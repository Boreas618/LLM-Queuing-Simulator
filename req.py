import numpy as np
from typing import List, Optional, Iterable, Dict
import pandas as pd
from dataclasses import dataclass
import json
from enum import Enum, auto


class RequestState(Enum):
    PREFILL_GLOBAL = auto()
    PREFILL_LOCAL = auto()
    DECODE_GLOBAL = auto()
    DECODE_LOCAL = auto()


@dataclass(slots=True)
class Request:
    idx: int
    arrival: float
    input_length: int
    output_length: int
    prefill_time: float | None = None
    decode_time: float | None = None
    prefill_start: float | None = None
    prefill_finish: float | None = None
    kv_transmission_finish: float | None = None
    decode_start: float | None = None
    decode_finish: float | None = None
    decode_steps: int = 0
    prefill_instance: int | None = None
    decode_instance: int | None = None

    def state(self):
        if self.prefill_instance is None:
            return RequestState.PREFILL_GLOBAL
        elif self.prefill_instance is not None and self.prefill_time is None:
            return RequestState.PREFILL_LOCAL
        elif self.kv_transmission_finish is not None and self.decode_instance is None:
            return RequestState.DECODE_GLOBAL
        elif self.decode_instance is not None and self.decode_start is None:
            return RequestState.DECODE_LOCAL
        else:
            raise RuntimeError("Invalid State")


class RequestFactory:
    def __init__(self, req_count: int, arrival_rate: float, max_len: Optional[int] = None):
        self._init_req_population(req_count, arrival_rate, max_len)

    def _init_req_population(self, req_count: int, arrival_rate: float, max_len: Optional[int]):
        self.arrival_rate = arrival_rate
        self.req_count = req_count

        self.inter = np.random.exponential(
            scale=1.0 / self.arrival_rate, size=self.req_count)
        self.arrivals = np.cumsum(self.inter)
        if max_len is None:
            self.max_len = 126195  # To fit Mooncake trace
        else:
            self.max_len = max_len

    def _post_process_lengths(self, lengths: Iterable[int]):
        interp_lengths = np.array(lengths).astype(float)
        len_mask = interp_lengths <= self.max_len
        interp_lengths[~len_mask] = np.nan
        interp_lengths = pd.Series(interp_lengths).interpolate(
            method='linear').fillna(method='bfill').fillna(method='ffill').to_numpy()
        return interp_lengths

    def _generate_zipfian(self, zipf_s: float) -> List[Request]:
        # TODO: the output length
        lengths = np.random.zipf(a=zipf_s, size=self.req_count)
        interp_lengths = self._post_process_lengths(lengths)
        return [Request(i, self.arrivals[i], int(interp_lengths[i]), int(interp_lengths[i] * 1.1)) for i in range(self.req_count)]

    def _generate_lognormal(self, lognormal_mean: float) -> List[Request]:
        # TODO: the output length
        rng = np.random.default_rng(2025)

        lengths = rng.lognormal(
            mean=lognormal_mean, sigma=1.0, size=self.req_count)
        interp_lengths = self._post_process_lengths(lengths)
        # return [Request(i, self.arrivals[i], int(interp_lengths[i]), int(interp_lengths[i] * 1.1)) for i in range(self.req_count)]
        return [Request(i, self.arrivals[i], int(interp_lengths[i]), 1) for i in range(self.req_count)]

    def _generate_mooncake(self, trace_path: str):
        raw_records: List[str] = []
        with open(trace_path) as f:
            raw_records = f.readlines()
        try:
            records = list(map(lambda r: json.loads(r), raw_records))
        except:
            print('failed to convert string to json.')

        input_lengths = np.array(
            list(map(lambda r: r['input_length'], records)))
        output_lengths = np.array(
            list(map(lambda r: r['output_length'], records)))

        self._init_req_population(
            len(input_lengths), self.arrival_rate, np.add(input_lengths, output_lengths).max())

        return [Request(i, self.arrivals[i], input_lengths[i], output_lengths[i]) for i in range(self.req_count)]

    def generate(self, source: str, param):
        if source == 'zipfian':
            return self._generate_zipfian(param)
        elif source == 'lognormal':
            return self._generate_lognormal(param)
        elif source == 'mooncake':
            return self._generate_mooncake(param)
        else:
            raise RuntimeError
