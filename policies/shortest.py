from .base import LocalPolicy
from req import Request
from simulator import SchedulingContext


class ShortestPolicy(LocalPolicy):
    identifier = 'shortest'

    def schedule(self, context: SchedulingContext) -> Request:
        if not context.queue:
            raise ValueError("Cannot schedule from empty queue")
        return min(context.queue, key=lambda r: r.input_length)
