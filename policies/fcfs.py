from .base import LocalPolicy
from req import Request
from simulator import SchedulingContext


class FCFSPolicy(LocalPolicy):
    identifier = 'fifo'

    def schedule(self, context: SchedulingContext) -> Request:
        if not context.queue:
            raise ValueError("Cannot schedule from empty queue")
        sorted_queue = sorted(context.queue, key=lambda r: r.arrival)
        return sorted_queue[0]
