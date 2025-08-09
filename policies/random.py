from .base import LocalPolicy, GlobalPolicy, LocalSchedContext, GlobalSchedContext
from typing import List
from req import Request
import random


class LocalRandomPolicy(LocalPolicy):
    identifier = 'random_local'

    def __init__(self):
        super().__init__()
        self.rng = random.Random()

    def schedule(self, queue: List[Request], context: LocalSchedContext) -> int:
        return self.rng.randrange(len(queue))


class GlobalRandomPolicy(GlobalPolicy):
    identifier = 'random_global'

    def __init__(self):
        super().__init__()
        self.rng = random.Random()

    def schedule(self, context: GlobalSchedContext):
        instances = context.instances()
        return self.rng.randrange(len(instances))
