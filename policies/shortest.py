from .base import LocalPolicy
from typing import List
from req import Request
from simulator import SchedContext


class ShortestPolicy(LocalPolicy):
    identifier = 'shortest'

    def schedule(self, queue: List[Request], _: SchedContext) -> int:
        return min(range(len(queue)), key=lambda i: queue[i].input_length)
