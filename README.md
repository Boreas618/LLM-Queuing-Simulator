# LLM-Serving-Simulator

LLM-Serving-Simulator is a simulator that models the complete lifecycle of request handling in LLM serving, including routing, queuing, prefill, KV cache transmission, and decoding. It is designed with extensibility in mind and facilitates easy verification of serving engine designs.

## Usage

Run the simulator with the following example command:

```bash
python main.py \
  --lambda_arrival 1.66 \
  --prefill_global lowest_load \
  --prefill_local shortest \
  --decode_global round_robin \
  --decode_local fcfs \
  --request_source mooncake \
  --trace_path traces/conversation_trace.jsonl \
  --seed 2025 \
  --ttft_slo 10.0 \
  --tpot_slo 0.5 \
  --sample_memory_usage \
  --sample_rate 1.0 \
  --queue_states_output queue_states.csv
```

## Acknowledgment

The profiling component is adapted from [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer), and the demonstration request trace is sourced from [Mooncake](https://github.com/kvcache-ai/Mooncake).

