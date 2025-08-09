from .base import GlobalPolicy, GlobalSchedContext
import itertools


class GlobalRoundRobinPolicy(GlobalPolicy):
    identifier = 'round_robin'

    def schedule(self, context: GlobalSchedContext):
        instances = context.instances()
        if self.counter is None:
            self.counter = itertools.cycle(range(len(instances)))
        return next(self.counter)
