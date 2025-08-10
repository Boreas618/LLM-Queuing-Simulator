from .base import LocalPolicy, GlobalPolicy
from req import Request
import random
from simulator import SchedulingContext, DispatchContext


class LocalRandomPolicy(LocalPolicy):
    identifier = 'random_local'

    def __init__(self):
        super().__init__()
        self.rng = random.Random()

    def schedule(self, context: SchedulingContext) -> Request:
        if not context.queue:
            raise ValueError("Cannot schedule from empty queue")
        return self.rng.choice(context.queue)


class GlobalRandomPolicy(GlobalPolicy):
    identifier = 'random_global'

    def __init__(self):
        super().__init__()
        self.rng = random.Random()

    def schedule(self, context: DispatchContext) -> int:
        request = context.request_to_dispatch
        if request.state().name.startswith('PREFILL'):
            instances = context.cluster_state.prefill_instances
        else:
            instances = context.cluster_state.decode_instances
        return self.rng.randrange(len(instances))
