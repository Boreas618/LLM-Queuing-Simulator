# LLM-Serving-Simulator

LLM-Serving-Simulator is a comprehensive event-driven simulator that models the complete lifecycle of request handling in LLM serving systems. It simulates routing, queuing, prefill processing, KV cache transmission, and decoding phases with realistic performance characteristics and memory constraints.

## Architecture Overview

### Core Components

The simulator consists of several key architectural components:

**Event-Driven Simulation Engine** (`simulator.py`): The heart of the system uses a priority queue-based event bus to process requests through their complete lifecycle. Events include request arrivals, prefill completions, KV cache transmissions, and decode steps.

**Plugin-Based Coordinator** (`coordinator.py`): Manages scheduling decisions through a flexible policy system. Loads scheduling policies dynamically from the `policies/` directory and handles both global (cross-instance) and local (within-instance) scheduling for prefill and decode phases.

**Extension System** (`extension.py`): Provides a unified framework for loading extensions including:
- **PolicyLoader**: Discovers and loads scheduling policies from the `policies/` directory
- **ModelConfigLoader**: Loads model configurations from the `models/` directory
- **ExtensionLoader**: Generic base class for discovering Python modules and classes

**Request Factory** (`req.py`): Generates request traces from multiple sources:
- `mooncake`: Real-world traces from JSONL files
- `zipfian`: Synthetic requests with Zipf distribution
- `lognormal`: Synthetic requests with log-normal distribution

### Extension System

The simulator uses a sophisticated plugin architecture that enables easy extensibility:

#### Scheduling Policies
All scheduling policies inherit from base classes in `policies/base.py`:
- **LocalPolicy**: Selects the next request from a queue (returns Request object)
- **GlobalPolicy**: Assigns requests to instances (returns instance ID)

The system makes four distinct scheduling decisions:
1. `prefill_global`: Which prefill instance receives new requests
2. `prefill_local`: Which queued request to process next on a prefill instance  
3. `decode_global`: Which decode instance receives requests after prefill
4. `decode_local`: Which queued requests to batch together for decode

Available policies include FCFS, shortest job first, lowest load, round robin, random, and MILP-based optimization.

#### Model Configurations
Model configurations are loaded dynamically from the `models/` directory, supporting different LLM architectures with their specific parameter counts, memory requirements, and performance characteristics.

### Instance Types and Request Lifecycle

The simulator models two types of compute instances:

**Prefill Instances**: Handle initial token processing and prompt evaluation. Configured with specific hardware profiles including memory limits, tensor parallelism settings, and compute capabilities.

**Decode Instances**: Handle autoregressive token generation with continuous batching. Support memory-aware scheduling with KV cache eviction when memory limits are exceeded.

Requests progress through these states:
1. `PREFILL_GLOBAL`: Waiting for prefill instance assignment
2. `PREFILL_LOCAL`: Queued on a prefill instance
3. `DECODE_GLOBAL`: Waiting for decode instance assignment  
4. `DECODE_LOCAL`: Queued/running on decode instance

### Parallel Processing and Checkpointing

The simulator supports large-scale experimentation through:

**Parallel Execution**: Uses multiprocessing to run parameter sweeps across multiple CPU cores. The `--n_jobs` parameter controls parallelism level.

**Checkpointing System**: Long-running experiments can be checkpointed and resumed:
- Results are serialized using pickle to `--checkpoint_path`
- Use `--resume` flag to continue from saved checkpoint
- Prevents loss of computation time for extensive parameter studies

### Configuration and Hardware Modeling

**Hardware Configurations**: 
- `configs/prefill_config.json`: Prefill instance specifications
- `configs/decode_config.json`: Decode instance specifications  
- `configs/hardware_params.py`: Hardware performance characteristics

**Realistic Performance Modeling**: Integrates with LLM-Viewer profiling data to provide accurate timing estimates for different model sizes, batch sizes, and hardware configurations.

### Metrics and Analysis

The simulator provides comprehensive metrics collection:

- **Performance Metrics**: TTFT (Time to First Token), TPOT (Time Per Output Token), throughput
- **SLO Compliance**: Tracks adherence to specified service level objectives
- **Queue States**: Optional sampling of queue lengths and states over time
- **Memory Usage**: Tracks memory consumption across decode instances
- **Visualization**: Built-in plotting capabilities for analysis

## Usage

### Quick Start

Run the simulator with the provided shell script:
```bash
./run.sh
```

### Full Command Line

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
  --queue_states_output queue_states.csv \
  --n_jobs 10 \
  --checkpoint_path results.pkl \
  --resume
```

### Key Parameters

- `--lambda_arrival`: Request arrival rate(s) in requests per second
- `--prefill_global/local`: Global and local prefill scheduling policies
- `--decode_global/local`: Global and local decode scheduling policies  
- `--request_source`: Source of request traces (mooncake, zipfian, lognormal)
- `--n_jobs`: Number of parallel processes for parameter sweeps
- `--resume`: Resume from checkpoint if available
- `--sample_*`: Enable detailed sampling of system states

## Acknowledgment

The profiling component is adapted from [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer), and the demonstration request trace is sourced from [Mooncake](https://github.com/kvcache-ai/Mooncake).

