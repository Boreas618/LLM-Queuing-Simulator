from abc import abstractmethod, ABC
from typing import List
from req import Request


class SchedContext:
    from simulator import Simulator

    def __init__(self, simulator: Simulator):
        self.simulator = simulator

    @property
    def time(self):
        return self.simulator.time()

    @property
    def requests(self):
        return self.simulator.requests()

    @property
    def request_stage(self):
        return self.simulator.request().stage

    @property
    def request_input_length(self):
        return self.simulator.request().input_length

    @property
    def request_output_length(self):
        """
        Gets the output length of the current request.

        Note: This should be used cautiously, as a scheduler is ideally
        agnostic to the true output length.
        """
        return self.simulator.request().output_length

    @property
    def ttft_slo(self):
        return self.simulator.ttft_slo()


class GlobalSchedContext(SchedContext):
    @property
    def instances(self):
        return self.simulator.instances()


class LocalSchedContext(SchedContext):
    @property
    def instance(self):
        return self.simulator.request().instance


class Policy(ABC):
    identifier: str
    pass


class LocalPolicy(Policy):
    @abstractmethod
    def schedule(self, queue: List[Request], context: LocalSchedContext) -> int:
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
    def schedule(self, context: GlobalSchedContext) -> int:
        pass
