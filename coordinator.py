from simulator import Simulator
from typing import List, Dict
import importlib
import inspect
from pathlib import Path
from policies.base import Policy, LocalPolicy, GlobalPolicy, SchedContext
import logging
from req import Request


class Coordinator:
    plugin_dir = 'policies'

    def __init__(self, prefill_local: str, prefill_global: str, decode_local: str, decode_global: str, ttft_slo: float, tpot_slo: float, sim: Simulator):
        self.prefill_local = prefill_local
        self.prefill_global = prefill_global
        self.decode_local = decode_local
        self.decode_global = decode_global
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self.sim = sim

        self.policies: Dict[str, Policy] = self._load_plugin()

        self.prefill_global_scheduler = self.policies[self.prefill_global]()
        self.prefill_local_scheduler = self.policies[self.prefill_local]()
        self.decode_global_scheduler = self.policies[self.decode_global]()
        self.decode_local_scheduler = self.policies[self.decode_local]()

    def _load_plugin(self):
        policies = {}
        plugin_path = Path(self.plugin_dir)

        for file in plugin_path.glob("*.py"):
            # Skip files like __init__.py
            if file.name.startswith("_"):
                continue

            # Convert file path to a module path (e.g., 'policies/fifo.py' -> 'policies.fifo')
            module_name = f"{plugin_path.name}.{file.stem}"

            try:
                module = importlib.import_module(module_name)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Policy) and obj is not Policy and obj is not LocalPolicy and obj is not GlobalPolicy:
                        policies[file.stem] = obj
                        logging.log(
                            logging.INFO, f"Discovered plugin: '{file.stem}'")

            except ImportError as e:
                logging.log(
                    logging.ERROR, f"Could not import plugin {file.name}: {e}'")

        return policies

    def prefill_dispatch(self, request: Request, context: SchedContext):
        return self.prefill_global_scheduler.schedule(request, context)

    def prefill_schedule(self, queue: List[Request], context: SchedContext) -> Request:
        index = self.prefill_local_scheduler.schedule(queue, context)
        request = queue.pop(index)
        return request

    def decode_dispatch(self, request: Request, context: SchedContext):
        return self.decode_global_scheduler.schedule(request, context)

    def decode_schedule(self, queue: List[Request], context: SchedContext):
        selected = []
        while context.current_instance.memory_usage < context.current_instance.memory_limit and len(queue) > 0:
            index = self.decode_local_scheduler.schedule(queue, context)
            selected.append(queue[index])
            context.current_instance.memory_usage += queue[index].input_length * \
                context.model_config.per_token_kv_cache_size_byte
            queue.pop(index)
        return selected
