from __future__ import annotations
import heapq
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple
from req import Request
import logging
from tqdm import tqdm
from configs.hardware_params import hardware_params


class EventIterator:
    def __init__(self, context: SchedContext):
        self.context = context

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.context.events) == 0:
            raise StopIteration
        t, ev_type, payload = heapq.heappop(self.context.events)

        self.context.set_time(t)

        current_request = None
        if ev_type == "arrival" or ev_type == "kv_transmission_finish":
            current_request = self.context.requests[int(payload)]

        current_instance = None
        if ev_type == "prefill_finish":
            current_instance = self.context.prefill_instances[int(payload)]
        elif ev_type == "decode_step":
            current_instance = self.context.decode_instances[int(payload)]

        assert current_request is not None or current_instance is not None
        self.context.set_current_event(current_request, current_instance)

        return t, ev_type, payload


class SchedContext:
    def __init__(
        self,
        model_config: ModelConfig,
        prefill_config: List[Tuple],
        decode_config: List[Tuple],
        ttft_slo: float,
        tpot_slo: float,
        requests: List[Request]
    ):
        self.model_config = model_config
        self.prefill_instances: List["ModelInstance"] = []
        self.decode_instances: List["ModelInstance"] = []

        for instance in prefill_config:
            hardware = instance["hardware"]
            tp_size = instance["tp_size"]
            memory_limit = hardware_params[instance["hardware"]]["mem"] - \
                self.model_config.model_size_byte / instance["tp_size"]
            self.prefill_instances.append(
                ModelInstance(hardware, tp_size, memory_limit)
            )

        for instance in decode_config:
            hardware = instance["hardware"]
            tp_size = instance["tp_size"]
            memory_limit = hardware_params[instance["hardware"]]["mem"] - \
                self.model_config.model_size_byte / instance["tp_size"]
            self.decode_instances.append(
                ModelInstance(hardware, tp_size, memory_limit)
            )

        for instance in self.prefill_instances:
            assert instance.memory_limit > 0
        for instance in self.decode_instances:
            assert instance.memory_limit > 0

        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self.requests = requests

        self.events: List[Tuple[float, str, int]] = [
            (req.arrival, "arrival", req.idx) for req in self.requests
        ]
        heapq.heapify(self.events)

        self.event_iterator = EventIterator(self)
        self.time = 0.0
        self.current_request = None
        self.current_instance = None
        self.current_state = None

    def set_time(self, t: float):
        self.time = t

    def set_current_event(self, request, instance):
        self.current_request = request
        self.current_instance = instance
        if self.current_request:
            self.current_state = self.current_request.state()

    def iter_events(self):
        return self.event_iterator

    def add_event(self, event: Tuple[float, str, int]):
        self.events.append(event)


@dataclass(slots=True)
class ModelInstance:
    hardware: str
    tp_size: int
    memory_limit: float

    busy: bool = False
    running_requests: List[Request] = field(default_factory=list)
    finish_time: float = 0.0
    queue: List[Request] = field(default_factory=list)
    memory_usage: float = 0


@dataclass(slots=True)
class ModelConfig:
    model_id: str
    weight_count: int
    w_bit: int = 16
    a_bit: int = 16
    kv_bit: int = 16
    use_flashattention: bool = True
    kv_cache_transmission_speed: float = 1e9
    per_token_kv_cache_size_byte: float = (16384 * 2 * 126 * 8 / 128) * 2

    @property
    def model_size_byte(self):
        return (self.weight_count * self.w_bit / 8) * (10 ** 9)


@dataclass(slots=True)
class QueueState:
    timestamp: float = 0.0
    instance_id: int = -1
    instance_group: str = ''
    queue: List[Request] = field(default_factory=list)


@dataclass(slots=True)
class MemoryUsageSample:
    timestamp: float = 0.0
    instance_id: int = -1
    instance_group: str = ''
    memory_usage: float = 0.0


class Simulator:
    def __init__(
        self,
        prefill_config: List[Tuple],
        decode_config: List[Tuple],
        prefill_global: str,
        prefill_local: str,
        decode_global: str,
        decode_local: str,
        ttft_slo: float,
        tpot_slo: float,
        requests: List[Request],
        model_config: ModelConfig,
        seed: int = 42,
        sample_queue_states: bool = False,
        sample_memory_usage: bool = False,
        sample_rate: float = 0.001,
    ):
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.context = SchedContext(
            model_config, prefill_config, decode_config, ttft_slo, tpot_slo, requests)

        self.sample_queue_states = sample_queue_states
        self.sample_memory_usage = sample_memory_usage
        self.sample_rate = sample_rate
        self.queue_states: List[QueueState] = []
        self.memory_usage_samples: List[MemoryUsageSample] = []

        from coordinator import Coordinator
        self.coordinator = Coordinator(
            prefill_local, prefill_global, decode_local, decode_global, ttft_slo, tpot_slo, self)

        from profiler import ModelAnalyzer
        self.analyzer = ModelAnalyzer(self.context.model_config.model_id)

        # Progress tracking
        self.max_decode_finished_id = 0
        self.total_requests = len(requests)
        self.progress_bar = tqdm(
            total=self.total_requests, desc="Progress", unit="event")

    def run(self) -> None:
        for t, ev_type, payload in self.context.iter_events():
            logging.info(f"{t} {ev_type} {payload}")
            instance: ModelInstance | None = None

            if ev_type == "arrival":
                req_id = payload
                req = self.context.requests[req_id]

                # Assign model instance
                assert req.input_length is not None
                instance_id = self.coordinator.prefill_dispatch(
                    req, self.context)
                req.prefill_instance = instance_id

                instance = self.context.prefill_instances[instance_id]
                assert instance is not None

                # TODO: batch size > 1 for prefill inference
                req.prefill_time = self.analyzer.analyze_prefill_iteration(
                    instance.hardware, req.input_length, 1, self.context.model_config.w_bit, self.context.model_config.a_bit, self.context.model_config.kv_bit, self.context.model_config.use_flashattention, instance.tp_size)

                if not instance.busy:
                    req.prefill_start = t
                    req.prefill_finish = t + req.prefill_time
                    instance.busy = True
                    instance.running_requests = [req]
                    instance.finish_time = req.prefill_finish
                    self.context.add_event(
                        (req.prefill_finish, "prefill_finish", instance_id))
                else:
                    # enqueue
                    instance.queue.append(req)

            elif ev_type == "prefill_finish":
                instance_id = payload
                instance = self.context.prefill_instances[instance_id]
                finished_request = instance.running_requests[0]
                assert finished_request is not None

                if finished_request.prefill_finish is None:
                    raise ValueError(
                        f"Prefill finish is None for request {finished_request.idx}")

                finished_request.kv_transmission_finish = finished_request.prefill_finish + finished_request.input_length * \
                    self.context.model_config.per_token_kv_cache_size_byte / \
                    self.context.model_config.kv_cache_transmission_speed

                self.context.add_event(
                    (finished_request.kv_transmission_finish, "kv_transmission_finish", finished_request.idx))

                instance.busy = False
                instance.running_requests.pop(0)

                # Dispatch next if any
                if instance.queue:
                    if self.sample_queue_states and self.rng.random() < self.sample_rate:
                        queue = [request for request in instance.queue]
                        self.queue_states.append(QueueState(
                            timestamp=t,
                            instance_id=instance_id,
                            instance_group='prefill',
                            queue=queue
                        ))

                    debug_len_before = len(instance.queue)
                    next_req = self.coordinator.prefill_schedule(
                        instance.queue, self.context)
                    debug_len_after = len(instance.queue)
                    assert debug_len_before == debug_len_after + \
                        1, f"{debug_len_before}, {debug_len_after}"

                    next_req.prefill_time = self.analyzer.analyze_prefill_iteration(
                        instance.hardware, next_req.input_length, 1, self.context.model_config.w_bit, self.context.model_config.a_bit, self.context.model_config.kv_bit, self.context.model_config.use_flashattention, instance.tp_size)

                    next_req.prefill_start = t
                    next_req.prefill_finish = t + next_req.prefill_time
                    instance.busy = True
                    instance.running_requests = [next_req]
                    instance.finish_time = next_req.prefill_finish
                    self.context.add_event(
                        (next_req.prefill_finish, "prefill_finish", instance_id))

            elif ev_type == "kv_transmission_finish":
                req_id = payload
                req = self.context.requests[req_id]

                instance_id = self.coordinator.decode_dispatch(
                    req, self.context)
                req.decode_instance = instance_id
                instance = self.context.decode_instances[instance_id]
                assert instance is not None

                if not instance.busy:
                    iteration_time = self.analyzer.analyze_decode_iteration(
                        instance.hardware, req.input_length, 1, self.context.model_config.w_bit, self.context.model_config.a_bit, self.context.model_config.kv_bit, self.context.model_config.use_flashattention, instance.tp_size
                    )
                    req.decode_start = t
                    instance.busy = True
                    instance.running_requests = [req]
                    self.context.add_event(
                        (t + iteration_time, "decode_step", instance_id))
                else:
                    # enqueue
                    instance.queue.append(req)
            elif ev_type == "decode_step":
                instance_id = payload
                instance = self.context.decode_instances[instance_id]

                # (continuous batching) first check if any requests have finished
                finished = []
                for index, request in enumerate(instance.running_requests):
                    request.decode_steps += 1
                    if request.decode_steps == request.output_length:
                        finished.append(index)
                        request.decode_finish = t

                        # Update progress bar
                        if request.idx > self.max_decode_finished_id:
                            self.progress_bar.update(
                                request.idx - self.max_decode_finished_id)
                            self.max_decode_finished_id = request.idx

                # (continuous batching) if found finished requests, pop them
                for index in sorted(finished, reverse=True):
                    instance.running_requests.pop(index)

                # update memory usage
                memory_usage = 0
                for request in instance.running_requests:
                    memory_usage += (request.input_length +
                                     request.decode_steps) * self.context.model_config.per_token_kv_cache_size_byte

                # sample memory usage
                if self.sample_memory_usage:
                    if self.rng.random() < self.sample_rate:
                        self.memory_usage_samples.append(MemoryUsageSample(
                            timestamp=t,
                            instance_id=instance_id,
                            instance_group='decode',
                            memory_usage=memory_usage / instance.memory_limit
                        ))

                # FIXME: If memory is saturated, evict all of them
                if memory_usage > instance.memory_limit:
                    for request in instance.running_requests:
                        request.input_length = request.input_length + request.decode_steps
                        request.output_length = request.output_length - request.decode_steps
                        request.decode_steps = 0
                        instance.queue.append(request)
                    instance.running_requests.clear()
                    instance.busy = False
                    instance.memory_usage = 0
                else:
                    instance.memory_usage = memory_usage

                # (continuous batching) add new requests to the batch
                if instance.queue:
                    next_requests = self.coordinator.decode_schedule(
                        instance.queue, self.context)

                    for request in next_requests:
                        request.decode_start = t
                        request.decode_instance = instance_id
                    instance.busy = True
                    instance.running_requests += next_requests

                if len(instance.running_requests) == 0:
                    instance.busy = False
                    continue

                padded_sequence_len = max(list(map(lambda request: (
                    request.input_length + request.decode_steps), instance.running_requests)))

                iteration_time = self.analyzer.analyze_decode_iteration(instance.hardware, padded_sequence_len, len(
                    instance.running_requests), self.context.model_config.w_bit, self.context.model_config.a_bit, self.context.model_config.kv_bit, self.context.model_config.use_flashattention, instance.tp_size)

                next_tick_time = t + iteration_time
                instance.finish_time = next_tick_time

                self.context.add_event((next_tick_time,
                                        "decode_step", instance_id))

        # Close progress bar when simulation is complete
        self.progress_bar.close()

    def request_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (
                    r.idx,
                    r.arrival,
                    r.input_length,
                    r.output_length,
                    r.prefill_start,
                    r.prefill_finish,
                    (r.prefill_finish or np.nan) - r.arrival,
                    r.decode_start,
                    r.decode_finish,
                    (r.decode_finish - r.kv_transmission_finish) /
                    r.output_length if r.decode_finish and r.decode_start else np.nan,
                )
                for r in self.context.requests
            ],
            columns=pd.Index([
                "id",
                "arrival",
                "input_length",
                "output_length",
                "prefill_start",
                "prefill_finish",
                "ttft",
                "decode_start",
                "decode_finish",
                "tpot",
            ]),
        )

    def queue_states_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (
                    q.timestamp,
                    q.instance_id,
                    q.instance_group,
                    [r.idx for r in q.queue],
                    [r.input_length for r in q.queue],
                    [r.arrival + self.context.ttft_slo -
                        q.timestamp for r in q.queue],
                )
                for q in self.context.queue_states
            ],
            columns=pd.Index([
                "timestamp",
                "instance_id",
                "instance_group",
                "indicies",
                "input_lengths",
                "deadlines"
            ]),
        )

    def memory_usage_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (
                    sample.timestamp,
                    sample.instance_id,
                    sample.instance_group,
                    sample.memory_usage
                )
                for sample in self.memory_usage_samples
            ],
            columns=pd.Index([
                "timestamp",
                "instance_id",
                "instance_group",
                "memory_usage"
            ]),
        )
