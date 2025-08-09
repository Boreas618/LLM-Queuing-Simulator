from .base import GlobalPolicy, GlobalSchedContext
from typing import List
from simulator import RequestStage, ModelInstance


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

    def schedule(self, context: GlobalSchedContext):
        if context.request_stage() != RequestStage.PREFILL:
            raise NotImplementedError()

        time = context.time()
        length = context.request_input_length()
        instances = context.instances()

        loads = self.compute_loads(time, length, instances)
        return min(enumerate(loads), key=lambda x: x[1])[0]
