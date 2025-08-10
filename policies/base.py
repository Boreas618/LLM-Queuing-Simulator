from abc import abstractmethod, ABC
from typing import TYPE_CHECKING
from req import Request

if TYPE_CHECKING:
    from simulator import DispatchContext, SchedulingContext


class Policy(ABC):
    identifier: str
    pass


class LocalPolicy(Policy):
    @abstractmethod
    def schedule(self, context: 'SchedulingContext') -> Request:
        """Selects the next request to schedule from the queue. 🧐

        This policy shouldn't assume the queue is sorted in any particular way,
        as other system policies might change its order.

        Args:
            context: The scheduling context, which provides a view of the system's
                current state and the queue to choose from.

        Returns:
            The selected request object. The simulator uses this to remove the
            request from the queue. This separation between making a decision
            (this function) and enforcing it (the simulator) is intentional and can
            be leveraged to improve system observability.
        """
        pass


class GlobalPolicy(Policy):
    @abstractmethod
    def schedule(self, context: 'DispatchContext') -> int:
        """Selects an instance for the request.

        Args:
            context: The dispatch context containing the request to be scheduled
                and the current state of the cluster.

        Returns:
            The instance ID where the request should be dispatched.
        """
        pass
