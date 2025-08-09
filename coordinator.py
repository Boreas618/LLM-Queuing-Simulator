from math import log
import random
from simulator import Simulator, ModelInstance
from typing import Callable, List, Optional, Tuple, Dict
import itertools
import importlib
import inspect
from pathlib import Path
from policies.base import Policy, LocalPolicy, GlobalPolicy
import logging


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

        self.policies: Dict[str, LocalPolicy |
                            GlobalPolicy] = self._load_plugin()

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

    def prefill_dispatch(self, t: float, length: Optional[int]):
        # TODO: schedule with prefill_global_scheduler
        return None

    def prefill_schedule(self, instance: ModelInstance, time: float):
         # TODO: schedule with prefill_local_scheduler
        return None

    def decode_dispatch(self, t: float, length: Optional[int]):
        # TODO: schedule with decode_global_scheduler
        return None

    def decode_schedule(self, instance: ModelInstance, time: float):
        queue = instance.queue

        selected = []
        while instance.memory_usage < instance.memory_limit and len(queue) > 0:
            # TODO: schedule with decode_local_scheduler
            selected.append(request_id)
            instance.memory_usage += self.sim.requests[request_id].input_length * \
                self.sim.per_token_kv_cache_size_byte

        for req_id in queue:
            self.decode_blocking_tbl[req_id][1].extend(selected)

        return selected
