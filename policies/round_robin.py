from .base import GlobalPolicy
import itertools
from simulator import DispatchContext
from req import RequestState


class GlobalRoundRobinPolicy(GlobalPolicy):
    identifier = 'round_robin'

    def __init__(self):
        super().__init__()
        self.counter = None

    def schedule(self, context: DispatchContext) -> int:
        request = context.request_to_dispatch
        if request.state() == RequestState.PREFILL_GLOBAL:
            instances = context.cluster_state.prefill_instances
        elif request.state() == RequestState.DECODE_GLOBAL:
            instances = context.cluster_state.decode_instances
        else:
            raise ValueError(
                f"Round robin policy not applicable for request state: {request.state()}")

        if self.counter is None:
            self.counter = itertools.cycle(range(len(instances)))
        return next(self.counter)
