from .base import GlobalPolicy
import itertools
from simulator import SchedContext
from req import Request, RequestState


class GlobalRoundRobinPolicy(GlobalPolicy):
    identifier = 'round_robin'

    def __init__(self):
        super().__init__()
        self.counter = None

    def schedule(self, _: Request, context: SchedContext):
        if context.current_state == RequestState.PREFILL_GLOBAL:
            instances = context.prefill_instances
        elif context.current_state == RequestState.DECODE_GLOBAL:
            instances = context.decode_instances

        if self.counter is None:
            self.counter = itertools.cycle(range(len(instances)))
        return next(self.counter)
