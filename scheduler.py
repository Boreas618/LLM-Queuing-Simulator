from math import log
import random
from simulator import Simulator, ModelInstance
from typing import Callable, List, Optional, Tuple
import itertools
from milp import max_slo_attainment


class Coordinator:
    def __init__(self, prefill_local: str, prefill_global: str, decode_local: str, decode_global: str, ttft_slo: float, tpot_slo: float, sim: Simulator):
        self.prefill_local = prefill_local
        self.prefill_global = prefill_global
        self.decode_local = decode_local
        self.decode_global = decode_global
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self.sim = sim
        self.prefill_blocking_tbl = {}
        self.decode_blocking_tbl = {}

        self.prefill_global_scheduler = self._create_prefill_global(
            len(sim.prefill_instances),
            self.sim.prefill_instances,
        )
        self.prefill_local_scheduler = self._create_prefill_local()
        self.decode_global_scheduler = self._create_decode_global(
            len(sim.decode_instances),
            self.sim.decode_instances,
        )
        self.decode_local_scheduler = self._create_decode_local()

    def alloc_blocking_tbl(self, table: str, key: int, value: Tuple[int, List[int]]):
        if table == "prefill":
            self.prefill_blocking_tbl[key] = value
        elif table == "decode":
            self.decode_blocking_tbl[key] = value
        else:
            raise ValueError(f"unknown table {table}")

    def _create_prefill_global(self, num_instances: int, instances: List["ModelInstance"]) -> Callable:
        rng = random.Random()

        def compute_loads(t: float, length: int, instances: List["ModelInstance"]) -> List[float]:
            loads = []
            for instance in instances:
                running_remaining = (
                    instance.finish_time - t) if instance.busy else 0.0
                queuing_time = sum(
                    [self.sim.requests[r].prefill_time if self.sim.requests[r].input_length < length else 0 for r in instance.queue])
                loads.append(running_remaining + queuing_time)
            return loads

        if self.prefill_global == "random":
            return lambda t, _: rng.randrange(num_instances)

        elif self.prefill_global == "round_robin":
            counter = itertools.cycle(range(num_instances))
            return lambda t, _: next(counter)

        elif self.prefill_global == "lowest_load":
            def choose_lowest_load(t: float, length: int) -> int:
                loads = compute_loads(t, length, instances)
                return min(enumerate(loads), key=lambda x: x[1])[0]
            return choose_lowest_load

        raise ValueError(f"unknown global policy '{self.prefill_global}'")

    def _create_decode_global(self, num_instances: int, instances: List["ModelInstance"]) -> Callable:
        rng = random.Random()

        if self.decode_global == "random":
            return lambda t, _: rng.randrange(num_instances)

        elif self.decode_global == "round_robin":
            counter = itertools.cycle(range(num_instances))
            return lambda t, _: next(counter)

        raise ValueError(f"unknown global policy '{self.decode_global}'")

    def _create_prefill_local(self) -> Callable:
        rng = random.Random()
        if self.prefill_local == "fcfs":
            return lambda _, __, ___: 0

        elif self.prefill_local == "random":
            return lambda queue, __, ___: rng.randrange(len(queue))

        elif self.prefill_local == "shortest":
            return lambda queue, __, ___: min(range(len(queue)), key=lambda i: self.sim.requests[queue[i]].input_length)

        elif self.prefill_local == "lottery":
            def choose(queue, instance, time):
                requests = list(
                    map(lambda req_id: self.sim.requests[req_id], queue))

                # Compute inverse weights
                weights = []
                for req in requests:
                    if req.input_length > 0:
                        weights.append(1 / req.input_length)
                    else:
                        weights.append(0)  # Avoid division by zero

                total_weight = sum(weights)
                if total_weight == 0:
                    # fallback: uniform random
                    return rng.randrange(len(queue))

                # Inverse-weighted lottery draw
                pick = rng.uniform(0, total_weight)
                cumulative = 0
                for i, weight in enumerate(weights):
                    cumulative += weight
                    if pick <= cumulative:
                        return i
            return choose

        elif self.prefill_local == "highest_priority":
            return lambda queue, __, ___: max(range(len(queue)), key=lambda i: self.sim.requests[queue[i]].priority)

        elif self.prefill_local == "hrrn":
            return lambda queue, _, t: max(range(len(queue)), key=lambda i: (log(t - self.sim.requests[queue[i]].arrival, 10) + self.sim.requests[queue[i]].prefill_time) / self.sim.requests[queue[i]].prefill_time)

        elif self.prefill_local == "milp":
            def choose_milp(queue, instance, t):
                requests = [self.sim.requests[i] for i in queue]
                filtered_requests = [(i, req) for i, req in enumerate(
                    requests) if req.arrival + self.sim.ttft_slo - t > 0]

                if len(filtered_requests) == 0:
                    # fallback to sjf
                    return min(range(len(queue)), key=lambda i: self.sim.requests[queue[i]].input_length)

                # TODO: questionable
                speed = sum(requests, key=lambda r: r.input_length /
                            r.prefill_time) / len(requests)

                index = max_slo_attainment(
                    [req.input_length for _, req in filtered_requests],
                    [req.arrival + self.sim.ttft_slo -
                        t for _, req in filtered_requests],
                    speed,
                )["schedule"][0]
                return filtered_requests[index][0]

            return choose_milp

        else:
            raise ValueError(f"unknown local policy '{self.prefill_local}'")

    def _create_decode_local(self) -> Callable:
        rng = random.Random()
        if self.decode_local == "fcfs":
            return lambda _, __, ___: 0
        else:
            raise ValueError(f"unknown local policy '{self.decode_local}'")

    def prefill_dispatch(self, t: float, length: Optional[int]):
        return self.prefill_global_scheduler(t, length)

    def prefill_schedule(self, instance: ModelInstance, time: float):
        queue = instance.queue
        k = self.prefill_local_scheduler(queue, instance, time)
        for req_id in queue:
            self.prefill_blocking_tbl[req_id][1].append(k)
        request_id = queue.pop(k)
        return request_id

    def decode_dispatch(self, t: float, length: Optional[int]):
        return self.decode_global_scheduler(t, length)

    def decode_schedule(self, instance: ModelInstance, time: float):
        queue = instance.queue

        selected = []
        while instance.memory_usage < instance.memory_limit and len(queue) > 0:
            k = self.decode_local_scheduler(queue, instance, time)
            request_id = queue.pop(k)
            selected.append(request_id)
            instance.memory_usage += self.sim.requests[request_id].input_length * \
                self.sim.per_token_kv_cache_size_byte

        for req_id in queue:
            self.decode_blocking_tbl[req_id][1].extend(selected)

        return selected
