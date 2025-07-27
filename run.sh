#!/usr/bin/env zsh

rm -rf figures

python main.py \
  --prefill_throughputs 10000.0 6000.0 4000.0 2000.0 1000.0 \
  --lambda_arrival 1.85 \
  --prefill_global lowest_load \
  --prefill_local shortest \
  --decode_global round_robin \
  --decode_local fcfs \
  --request_source mooncake \
  --trace_path conversation_trace.jsonl \
  --seed 2025 \
  --ttft_slo 10.0 \
  --tpot_slo 0.5 \
  --sample_queue_states \
  --sample_rate 1.0 \
  --queue_states_output queue_states.csv \
  --sample_memory_usage \
  --memory_usage_output memory_usage.csv \
  # --resume \
  # --checkpoint_path results.pkl

