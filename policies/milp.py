import itertools
from typing import List, Dict, Any
from simulator import SchedContext

from pulp import (
    LpProblem, LpMaximize, LpVariable, LpBinary, LpStatus,
    lpSum, value, PULP_CBC_CMD
)

from .base import LocalPolicy, SchedContext
from req import Request


class MILPPolicy(LocalPolicy):
    """
    A scheduling policy that uses Mixed-Integer Linear Programming (MILP)
    to maximize Service Level Objective (SLO) attainment.
    """

    identifier = 'milp'

    @staticmethod
    def max_slo_attainment(
        proc_times: List[float],
        deadlines: List[float],
        solver=None
    ) -> Dict[str, Any]:
        """
        Finds an optimal schedule to maximize the number of tasks meeting their deadline.

        Args:
            proc_times: A list of processing times for each task, $L_i$.
            deadlines: A list of deadlines for each task, $d_i$.
            solver: An optional PuLP-compatible solver.

        Returns:
            A dictionary containing the solution status, SLO attainment rate,
            the optimal schedule, and the corresponding finish times.
        """
        num_tasks = len(proc_times)
        if len(deadlines) != num_tasks:
            raise ValueError(
                "Length of processing times and deadlines must be the same.")

        # A sufficiently large number 'M' for the "Big-M" formulation.
        # The sum of all processing times is a safe upper bound for any single task's
        # completion time in a valid schedule.
        big_m = sum(proc_times)

        prob = LpProblem("SLO_Attainment_Maximization", LpMaximize)

        # y_i = 1 if task i meets its SLO, 0 otherwise
        meets_slo = [LpVariable(
            f"meets_slo_{i}", cat=LpBinary) for i in range(num_tasks)]

        # C_i = completion time of task i
        completion_times = [LpVariable(
            f"C_{i}", lowBound=0) for i in range(num_tasks)]

        # z_ij = 1 if task i is scheduled *after* task j, 0 otherwise
        is_scheduled_after = [
            [LpVariable(f"z_{i}_{j}", cat=LpBinary) for j in range(num_tasks)]
            for i in range(num_tasks)
        ]

        # Maximize the total number of tasks that meet their SLO.
        prob += lpSum(meets_slo), "maximize_slo_attainment"

        for i in range(num_tasks):
            # Constraint: A task's completion time must be at least its own processing time.
            prob += completion_times[i] >= proc_times[i], f"min_completion_time_{i}"

            # Constraint: Link completion time to the SLO deadline using the Big-M method.
            # If meets_slo[i] is 1, then completion_times[i] <= deadlines[i].
            # If meets_slo[i] is 0, the constraint is relaxed: completion_times[i] <= deadlines[i] + M.
            prob += completion_times[i] <= deadlines[i] + \
                big_m * (1 - meets_slo[i]), f"slo_check_{i}"

        for i, j in itertools.permutations(range(num_tasks), 2):
            # Constraint: Enforce a total ordering. For any pair of tasks (i, j),
            # either i is scheduled after j or j is scheduled after i.
            prob += is_scheduled_after[i][j] + \
                is_scheduled_after[j][i] == 1, f"total_order_{i}_{j}"

            # Constraint: Ensure correct time propagation using the Big-M method.
            # If task i is scheduled after task j (z_ij = 1), then its completion time must be
            # at least the completion time of j plus its own processing time.
            # C_i >= C_j + L_i
            prob += (
                completion_times[i] >= completion_times[j] +
                proc_times[i] - big_m * (1 - is_scheduled_after[i][j]),
                f"time_propagation_{i}_{j}"
            )

        solver = solver or PULP_CBC_CMD(msg=False)  # Use CBC solver by default
        prob.solve(solver)

        if prob.status != 1:  # 1 means "Optimal"
            return {
                "status": LpStatus[prob.status],
                "attainment": 0.0,
                "schedule": [],
                "finish_time": [],
            }

        # Create a list of (original_index, finish_time) and sort to get the schedule
        solved_schedule = sorted(
            [(i, value(completion_times[i])) for i in range(num_tasks)],
            key=lambda item: item[1]
        )

        return {
            "status": LpStatus[prob.status],
            "attainment": sum(value(v) for v in meets_slo) / num_tasks,
            "schedule": [idx for idx, _ in solved_schedule],
            "finish_time": [finish for _, finish in solved_schedule],
        }

    def schedule(self, queue: List[Request], context: SchedContext) -> int:
        """
        Determines the next request to schedule from the queue.
        """
        current_time = context.time
        slo = context.ttft_slo

        # Filter for requests whose deadlines have not yet passed
        schedulable_reqs = [
            (i, req) for i, req in enumerate(queue)
            if (req.arrival + slo) > current_time
        ]

        # If no requests can possibly meet their SLO, fall back to SJF
        if not schedulable_reqs:
            return min(range(len(queue)), key=lambda i: queue[i].input_length)

        proc_times = [context.instance.analyzer(
            req.input_length) for _, req in schedulable_reqs]
        deadlines = [(req.arrival + slo - current_time)
                     for _, req in schedulable_reqs]

        result = self.max_slo_attainment(proc_times, deadlines)

        # This policy is a "re-planning" policy. At each step, it computes the
        # full optimal schedule but only dispatches the *first* task from it.
        first_task_idx = result["schedule"][0]

        # Map the index from the filtered list back to the original queue index
        index = schedulable_reqs[first_task_idx][0]
        return index
