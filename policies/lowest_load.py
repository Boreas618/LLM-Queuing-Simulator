from .base import GlobalPolicy
from typing import List
from simulator import ModelInstance
from req import RequestState
from simulator import DispatchContext


class GlobalLowestLoadPolicy(GlobalPolicy):
    identifier = 'lowest_load'

    def compute_loads(self, time: float, length: int, instances: List[ModelInstance]) -> List[float]:
        loads = []
        for instance in instances:
            running_remaining = (
                instance.finish_time - time) if instance.busy else 0.0
            queuing_time = sum(
                [r.prefill_time if r.input_length < length else 0 for r in instance.queue])
            loads.append(running_remaining + queuing_time)
        return loads

    def schedule(self, context: DispatchContext) -> int:
        request = context.request_to_dispatch
        if request.state() != RequestState.PREFILL_GLOBAL:
            raise NotImplementedError(
                "lowest load policy only supports prefill stage now")

        time = context.current_time
        length = request.input_length
        instances = context.cluster_state.prefill_instances

        loads = self.compute_loads(time, length, instances)
        return min(enumerate(loads), key=lambda x: x[1])[0]
