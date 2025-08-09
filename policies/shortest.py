from .base import LocalPolicy, LocalSchedContext
from typing import List
from req import Request


class ShortestPolicy(LocalPolicy):
    identifier = 'shortest'

    def schedule(self, queue: List[Request], context: LocalSchedContext) -> int:
        return min(range(len(queue)), key=lambda i: queue[i].input_length)
