from .base import LocalPolicy
from simulator import SchedContext
from typing import List
from req import Request


class FCFSPolicy(LocalPolicy):
    identifier = 'fifo'

    def schedule(self, queue: List[Request], _: SchedContext) -> int:
        sorted(queue, key=lambda r: r.arrival)
        return 0
