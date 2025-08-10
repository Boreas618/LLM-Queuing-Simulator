from .base import LocalPolicy, GlobalPolicy
from typing import List
from req import Request
import random
from simulator import SchedContext


class LocalRandomPolicy(LocalPolicy):
    identifier = 'random_local'

    def __init__(self):
        super().__init__()
        self.rng = random.Random()

    def schedule(self, queue: List[Request], context: SchedContext) -> int:
        return self.rng.randrange(len(queue))


class GlobalRandomPolicy(GlobalPolicy):
    identifier = 'random_global'

    def __init__(self):
        super().__init__()
        self.rng = random.Random()

    def schedule(self, _: Request, context: SchedContext):
        instances = context.instances()
        return self.rng.randrange(len(instances))
