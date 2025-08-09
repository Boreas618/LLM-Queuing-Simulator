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
from enum import Enum, auto


class RequestStage(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclass(slots=True)
class ModelInstance:
    hardware: str
    tp_size: int
    memory_limit: float

    busy: bool = False
    # Index of the request being processed
    running_id: List[int] = field(default_factory=list)
    # The finish time of the current running request
    finish_time: float = 0.0
    queue: List[int] = field(default_factory=list)
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

        self.model_config = model_config

        self.prefill_instances: List["ModelInstance"] = [
            ModelInstance(hardware=instance["hardware"], tp_size=instance["tp_size"], memory_limit=hardware_params[instance["hardware"]]["mem"] - self.model_config.model_size_byte / instance["tp_size"]) for instance in prefill_config]
        self.decode_instances: List["ModelInstance"] = [
            ModelInstance(hardware=instance["hardware"], tp_size=instance["tp_size"], memory_limit=(hardware_params[instance["hardware"]]["mem"] - (self.model_config.model_size_byte / instance["tp_size"]))) for instance in decode_config]

        for instance in self.prefill_instances:
            assert instance.memory_limit > 0
        for instance in self.decode_instances:
            assert instance.memory_limit > 0

        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self.requests = requests

        self.sample_queue_states = sample_queue_states
        self.sample_memory_usage = sample_memory_usage
        self.sample_rate = sample_rate
        self.queue_states: List[QueueState] = []
        self.memory_usage_samples: List[MemoryUsageSample] = []

        from coordinator import Coordinator
        self.coordinator = Coordinator(
            prefill_local, prefill_global, decode_local, decode_global, ttft_slo, tpot_slo, self)

        from profiler import ModelAnalyzer
        self.analyzer = ModelAnalyzer(self.model_config.model_id)

        # event-queue : (time, type, payload)
        self.events: List[Tuple[float, str, int]] = [
            (req.arrival, "arrival", req.idx) for req in self.requests
        ]
        heapq.heapify(self.events)

        # TODO: refactor
        self.per_token_kv_cache_size_byte = self.model_config.per_token_kv_cache_size_byte
        self.kv_cache_transmission_speed = self.model_config.kv_cache_transmission_speed

        # Progress tracking
        self.max_decode_finished_id = 0
        self.total_requests = len(requests)
        self.progress_bar = tqdm(
            total=self.total_requests, desc="Progress", unit="event")

    def run(self) -> None:
        while self.events:
            t, ev_type, payload = heapq.heappop(self.events)
            logging.info(f"{t} {ev_type} {payload}")
            instance: ModelInstance | None = None

            if ev_type == "arrival":
                req_id = payload
                req = self.requests[req_id]

                # Assign model instance
                assert req.input_length is not None
                instance_id = self.coordinator.prefill_dispatch(
                    t, req.input_length)
                req.instance = instance_id

                instance = self.prefill_instances[instance_id]
                assert instance is not None

                # TODO: batch size > 1 for prefill inference
                req.prefill_time = self.analyzer.analyze_prefill_iteration(
                    instance.hardware, req.input_length, 1, self.model_config.w_bit, self.model_config.a_bit, self.model_config.kv_bit, self.model_config.use_flashattention, instance.tp_size)

                self.coordinator.alloc_blocking_tbl(
                    "prefill", req_id, (req.input_length, []))

                if not instance.busy:
                    req.prefill_start = t
                    req.prefill_finish = t + req.prefill_time
                    instance.busy = True
                    instance.running_id = [req_id]
                    instance.finish_time = req.prefill_finish
                    heapq.heappush(
                        self.events, (req.prefill_finish, "prefill_finish", instance_id))
                else:
                    # enqueue
                    instance.queue.append(req_id)

            elif ev_type == "prefill_finish":
                instance_id = payload
                instance = self.prefill_instances[instance_id]
                finished_req_id = instance.running_id[0]
                assert finished_req_id is not None

                # TODO: heterogeneous interconnect
                req = self.requests[finished_req_id]
                if req.prefill_finish is None:
                    raise ValueError(
                        f"Prefill finish is None for request {req.idx}")

                req.kv_transmission_finish = req.prefill_finish + req.input_length * \
                    self.per_token_kv_cache_size_byte / self.kv_cache_transmission_speed
                heapq.heappush(
                    self.events, (req.kv_transmission_finish, "kv_transmission_finish", req.idx))

                instance.busy = False
                instance.running_id.clear()

                # Dispatch next if any
                if instance.queue:
                    if self.sample_queue_states and self.rng.random() < self.sample_rate:
                        queue = [self.requests[req_id]
                                 for req_id in instance.queue]
                        self.queue_states.append(QueueState(
                            timestamp=t,
                            instance_id=instance_id,
                            instance_group='prefill',
                            queue=queue
                        ))

                    next_id = self.coordinator.prefill_schedule(instance, t)
                    next_req = self.requests[next_id]

                    next_req.prefill_time = self.analyzer.analyze_prefill_iteration(
                        instance.hardware, next_req.input_length, 1, self.model_config.w_bit, self.model_config.a_bit, self.model_config.kv_bit, self.model_config.use_flashattention, instance.tp_size)

                    next_req.prefill_start = t
                    next_req.prefill_finish = t + next_req.prefill_time
                    instance.busy = True
                    instance.running_id = [next_id]
                    instance.finish_time = next_req.prefill_finish
                    heapq.heappush(
                        self.events, (next_req.prefill_finish, "prefill_finish", instance_id))

            elif ev_type == "kv_transmission_finish":
                req_id = payload
                req = self.requests[req_id]

                instance_id = self.coordinator.decode_dispatch(t, None)
                req.instance = instance_id
                instance = self.decode_instances[instance_id]
                assert instance is not None

                self.coordinator.alloc_blocking_tbl(
                    "decode", req_id, (req.output_length, []))

                if not instance.busy:
                    iteration_time = self.analyzer.analyze_decode_iteration(
                        instance.hardware, req.input_length, 1, self.model_config.w_bit, self.model_config.a_bit, self.model_config.kv_bit, self.model_config.use_flashattention, instance.tp_size
                    )
                    req.decode_start = t
                    instance.busy = True
                    instance.running_id = [req_id]
                    heapq.heappush(
                        self.events, (t + iteration_time, "decode_step", instance_id))
                else:
                    # enqueue
                    instance.queue.append(req_id)
            elif ev_type == "decode_step":
                instance_id = payload
                instance = self.decode_instances[instance_id]

                # (continuous batching) first check if any requests have finished
                finished = []
                for index, req_id in enumerate(instance.running_id):
                    req = self.requests[req_id]
                    req.decode_steps += 1
                    if req.decode_steps == req.output_length:
                        finished.append(index)
                        req.decode_finish = t

                        # Update progress bar
                        if req_id > self.max_decode_finished_id:
                            self.progress_bar.update(
                                req_id - self.max_decode_finished_id)
                            self.max_decode_finished_id = req_id

                # (continuous batching) if found finished requests, pop them
                for index in sorted(finished, reverse=True):
                    instance.running_id.pop(index)

                # update memory usage
                memory_usage = 0
                for req_id in instance.running_id:
                    req = self.requests[req_id]
                    memory_usage += (req.input_length +
                                     req.decode_steps) * self.per_token_kv_cache_size_byte

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
                    for req_id in instance.running_id:
                        req = self.requests[req_id]
                        req.input_length = req.input_length + req.decode_steps
                        req.output_length = req.output_length - req.decode_steps
                        req.decode_steps = 0
                        instance.queue.append(req_id)
                    instance.running_id.clear()
                    instance.busy = False
                    instance.memory_usage = 0
                else:
                    instance.memory_usage = memory_usage

                # (continuous batching) add new requests to the batch
                if instance.queue:
                    next_ids = self.coordinator.decode_schedule(instance, t)

                    for next_id in next_ids:
                        next_req = self.requests[next_id]
                        next_req.decode_start = t
                        next_req.instance = instance_id
                    instance.busy = True
                    instance.running_id += next_ids

                if len(instance.running_id) == 0:
                    instance.busy = False
                    continue

                # TODO: Is it valid to pad sequences with different lengths?
                padded_sequence_len = max(list(map(lambda id: (
                    self.requests[id].input_length + self.requests[id].decode_steps), instance.running_id)))

                iteration_time = self.analyzer.analyze_decode_iteration(instance.hardware, padded_sequence_len, len(
                    instance.running_id), self.model_config.w_bit, self.model_config.a_bit, self.model_config.kv_bit, self.model_config.use_flashattention, instance.tp_size)

                next_tick_time = t + iteration_time
                instance.finish_time = next_tick_time
                heapq.heappush(self.events, (next_tick_time,
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
                for r in self.requests
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
                    [r.arrival + self.ttft_slo - q.timestamp for r in q.queue],
                )
                for q in self.queue_states
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
