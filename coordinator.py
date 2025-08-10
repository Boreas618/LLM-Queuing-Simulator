from simulator import Simulator, DispatchContext, SchedulingContext, ClusterState
from typing import List, Dict
from policies.base import Policy
import logging
from req import Request
from extension import PolicyLoader


class Coordinator:

    def __init__(self, prefill_local: str, prefill_global: str, decode_local: str, decode_global: str, ttft_slo: float, tpot_slo: float, sim: Simulator):
        self.prefill_local = prefill_local
        self.prefill_global = prefill_global
        self.decode_local = decode_local
        self.decode_global = decode_global
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self.sim = sim

        # Use unified extension loader for policies
        policy_loader = PolicyLoader(logger=logging.getLogger(__name__))
        self.policies: Dict[str, Policy] = policy_loader.load_extensions()

        self.prefill_global_scheduler = self.policies[self.prefill_global]()
        self.prefill_local_scheduler = self.policies[self.prefill_local]()
        self.decode_global_scheduler = self.policies[self.decode_global]()
        self.decode_local_scheduler = self.policies[self.decode_local]()

    def prefill_dispatch(self, request: Request, cluster_state: ClusterState, time: float) -> int:
        # Create a specific, immutable context for this decision
        context = DispatchContext(
            current_time=time,
            cluster_state=cluster_state,
            request_to_dispatch=request
        )
        return self.prefill_global_scheduler.schedule(context)

    def prefill_schedule(self, queue: List[Request], instance_id: int, cluster_state: ClusterState, time: float) -> Request:
        context = SchedulingContext(
            current_time=time,
            cluster_state=cluster_state,
            queue=tuple(queue),
            instance_id=instance_id
        )
        # Policy now returns the selected request object directly, not an index
        selected_request = self.prefill_local_scheduler.schedule(context)
        # Remove the selected request from the queue
        queue.remove(selected_request)
        return selected_request

    def decode_dispatch(self, request: Request, cluster_state: ClusterState, time: float) -> int:
        context = DispatchContext(
            current_time=time,
            cluster_state=cluster_state,
            request_to_dispatch=request
        )
        return self.decode_global_scheduler.schedule(context)

    def decode_schedule(self, queue: List[Request], instance_id: int, cluster_state: ClusterState, time: float) -> List[Request]:
        selected = []
        instance = cluster_state.get_decode_instance(instance_id)

        while instance.memory_usage < instance.memory_limit and len(queue) > 0:
            context = SchedulingContext(
                current_time=time,
                cluster_state=cluster_state,
                queue=tuple(queue),
                instance_id=instance_id
            )
            selected_request = self.decode_local_scheduler.schedule(context)
            selected.append(selected_request)
            instance.memory_usage += selected_request.input_length * \
                cluster_state.model_config.per_token_kv_cache_size_byte
            queue.remove(selected_request)
        return selected
