from abc import abstractmethod, ABC
from typing import List
from req import Request
from simulator import SchedContext

class Policy(ABC):
    identifier: str
    pass


class LocalPolicy(Policy):
    @abstractmethod
    def schedule(self, queue: List[Request], context: SchedContext) -> int:
        """Selects the next request to schedule from the queue. 🧐

        This policy shouldn't assume the queue is sorted in any particular way,
        as other system policies might change its order.

        Args:
            queue: A list of request objects waiting to be scheduled. Using objects
                instead of IDs allows policies to easily access request attributes.
            context: The scheduling context, which provides a view of the system's
                current state.

        Returns:
            The index of the selected request in the queue. The simulator uses this
            index to remove the request. This separation between making a decision
            (this function) and enforcing it (the simulator) is intentional and can
            be leveraged to improve system observability.
        """
        pass


class GlobalPolicy(Policy):
    @abstractmethod
    def schedule(self, request: Request, context: SchedContext) -> int:
        pass
