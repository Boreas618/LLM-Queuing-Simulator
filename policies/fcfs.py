from .base import LocalPolicy, LocalSchedContext
from typing import List
from req import Request


class FCFSPolicy(LocalPolicy):
    identifier = 'fifo'

    def schedule(self, queue: List[Request], _: LocalSchedContext) -> int:
        sorted(queue, lambda r: r.arrival)
        return 0
