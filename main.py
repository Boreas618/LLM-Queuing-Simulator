from __future__ import annotations
from typing import List, Dict
import logging
from simulator import Simulator, ModelConfig
import argparse
from req import Request, RequestFactory
import pandas as pd
import itertools
import multiprocessing as mp
from functools import partial
from visualiation import TTFTVisualizer
import pickle
import json

logging.basicConfig(
    filename='events.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulation parameters for throughput and scheduling strategies.")

    parser.add_argument(
        "--prefill_config",
        type=str,
        default='configs/prefill_config.json',
        help="Path to a JSON file containing the configuration for the prefill instances"
    )
    parser.add_argument(
        "--decode_config",
        type=str,
        default='configs/decode_config.json',
        help="Path to a JSON file containing the configuration for the decode instances"
    )
    parser.add_argument(
        "--request_count",
        type=int,
        default=10000,
        help="Total number of requests."
    )
    parser.add_argument(
        "--lambda_arrival",
        type=float,
        nargs='+',
        default=[2.0],
        help="Request arrival rate(s) (requests per second). Can specify multiple values."
    )
    parser.add_argument(
        "--prefill_global",
        type=str,
        nargs='+',
        choices=["random", "lowest_load", "round_robin"],
        default=["lowest_load"],
        help="Global prefill scheduling strategies. Can specify multiple values."
    )
    parser.add_argument(
        "--prefill_local",
        type=str,
        nargs='+',
        choices=["fcfs", "shortest", "random", "hrrn", "milp"],
        default=["fcfs"],
        help="Local prefill scheduling strategies. Can specify multiple values."
    )
    parser.add_argument(
        "--decode_global",
        type=str,
        nargs='+',
        choices=["random", "lowest_load", "round_robin"],
        default=["lowest_load"],
        help="Global decode scheduling strategies. Can specify multiple values."
    )
    parser.add_argument(
        "--decode_local",
        type=str,
        nargs='+',
        choices=["fcfs"],
        default=["fcfs"],
        help="Local decode scheduling strategies. Can specify multiple values."
    )
    parser.add_argument(
        "--request_source",
        type=str,
        choices=["mooncake", "zipfian", "lognormal"],
        default=["mooncake"],
        help="Source of request trace."
    )
    parser.add_argument(
        "--trace_path",
        type=str,
        default="traces/conversation_trace.jsonl",
        help="Path to the Mooncake trace file (JSON Lines format). Used only if request_source is 'mooncake'."
    )
    parser.add_argument(
        "--zipf_s",
        type=float,
        default=2.0,
        help="Zipf distribution parameter. Used only if request_source is 'zipfian'."
    )
    parser.add_argument(
        "--lognormal_mean",
        type=float,
        default=9.0,
        help="Mean of the lognormal distribution. Used only if request_source is 'lognormal'."
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Optional maximum sequence length for synthetic generators."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed."
    )
    parser.add_argument(
        "--ttft_slo",
        type=float,
        nargs='+',
        default=[10.0],
        help="TTFT SLO(s) (seconds). Can specify multiple values."
    )
    parser.add_argument(
        "--tpot_slo",
        type=float,
        nargs='+',
        default=[0.5],
        help="TPOT SLO(s) (seconds). Can specify multiple values."
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default='results.csv',
        help="Optional path to save statistics to a CSV file."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=10,
        help="Number of parallel processes to use. Defaults to number of CPU cores."
    )
    parser.add_argument(
        "--resume",
        action='store_true',
        help="Resume from the checkpoint if it exists."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default='results.pkl',
        help="Path to the checkpoint the simulation results. Since the simulation is time-consuming, we can save the results and recover later."
    )
    parser.add_argument(
        "--sample_queue_states",
        action='store_true',
        help="Sample the queue states of prefill instances."
    )
    parser.add_argument(
        "--sample_memory_usage",
        action='store_true',
        help="Sample the memory usage of the decode instances."
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.001,
        help="Sample rate for the queue states of prefill instances."
    )
    parser.add_argument(
        "--queue_states_output",
        type=str,
        default='queue_states.csv',
        help="Path to save the queue states to a CSV file."
    )
    parser.add_argument(
        "--memory_usage_output",
        type=str,
        default='memory_usage.csv',
        help="Path to save the memory usage to a CSV file."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default='unsloth/Llama-3.3-70B-Instruct',
        help="The huggingface id of the model to be served"
    )
    return parser.parse_args()


def create_statistics_df(df, prefill_global, prefill_local, decode_global, decode_local,
                         ttft_slo, tpot_slo, request_count, lambda_arrival, request_source,
                         prefill_config, decode_config, seed):
    """Create a DataFrame with simulation statistics for CSV output."""
    stats = {
        'uuid': [f"{prefill_global}-{prefill_local}-{decode_global}-{decode_local}-{lambda_arrival}"],
        'prefill_global': [prefill_global],
        'prefill_local': [prefill_local],
        'decode_global': [decode_global],
        'decode_local': [decode_local],
        'ttft_slo': [ttft_slo],
        'tpot_slo': [tpot_slo],
        'request_count': [request_count],
        'lambda_arrival': [lambda_arrival],
        'request_source': [request_source],
        'prefill_config': [prefill_config],
        'decode_config': [decode_config],
        'seed': [seed],
        'ttft_p25': [df['ttft'].quantile(0.25)],
        'ttft_p50': [df['ttft'].quantile(0.50)],
        'ttft_p75': [df['ttft'].quantile(0.75)],
        'ttft_p90': [df['ttft'].quantile(0.90)],
        'ttft_p95': [df['ttft'].quantile(0.95)],
        'ttft_p99': [df['ttft'].quantile(0.99)],
        'ttft_mean': [df['ttft'].mean()],
        'ttft_std': [df['ttft'].std()],
        'slo_attainment_pct': [len(df[df['ttft'] < ttft_slo]) / len(df) * 100],
        'total_requests': [len(df)]
    }

    # Add TPOT statistics if the column exists
    if 'tpot' in df.columns:
        stats.update({
            'tpot_p25': [df['tpot'].quantile(0.25)],
            'tpot_p50': [df['tpot'].quantile(0.50)],
            'tpot_p75': [df['tpot'].quantile(0.75)],
            'tpot_p90': [df['tpot'].quantile(0.90)],
            'tpot_p95': [df['tpot'].quantile(0.95)],
            'tpot_p99': [df['tpot'].quantile(0.99)],
            'tpot_mean': [df['tpot'].mean()],
            'tpot_std': [df['tpot'].std()],
            'tpot_slo_attainment_pct': [len(df[df['tpot'] < tpot_slo]) / len(df) * 100]
        })

    return pd.DataFrame(stats)


def run_single_simulation(params, request_count, request_source, request_param,
                          prefill_config, decode_config, seed, model_id,
                          sample_queue_states, sample_memory_usage, sample_rate, queue_states_output, memory_usage_output):
    """Run a single simulation with given parameters."""
    lambda_arrival, prefill_global, prefill_local, decode_global, decode_local, ttft_slo, tpot_slo = params

    # Generate requests for this parameter combination
    request_factory = RequestFactory(request_count, lambda_arrival)
    requests: List[Request] = request_factory.generate(
        request_source, request_param)
    model_config: ModelConfig = ModelConfig(model_id=model_id, weight_count=70)

    # Run simulation
    sim = Simulator(
        prefill_config,
        decode_config,
        prefill_global,
        prefill_local,
        decode_global,
        decode_local,
        ttft_slo,
        tpot_slo,
        requests,
        model_config,
        seed,
        sample_queue_states=sample_queue_states,
        sample_memory_usage=sample_memory_usage,
        sample_rate=sample_rate,
    )
    sim.run()
    df = sim.request_df()
    if sample_queue_states:
        queue_states_df = sim.queue_states_df()
        queue_states_df.to_csv(queue_states_output, index=False)
    if sample_memory_usage:
        memory_usage_df = sim.memory_usage_df()
        memory_usage_df.to_csv(memory_usage_output, index=False)

    # Create statistics DataFrame
    stats_df = create_statistics_df(
        df, prefill_global, prefill_local, decode_global, decode_local,
        ttft_slo, tpot_slo, request_count, lambda_arrival, request_source,
        str(prefill_config), str(decode_config), seed
    )

    return df, stats_df, params


if __name__ == "__main__":
    args = parse_args()

    request_count = args.request_count
    request_source = args.request_source
    seed = args.seed

    # Prepare parameter lists for iteration
    lambda_arrivals = args.lambda_arrival
    prefill_globals = args.prefill_global
    prefill_locals = args.prefill_local
    decode_globals = args.decode_global
    decode_locals = args.decode_local
    ttft_slos = args.ttft_slo
    tpot_slos = args.tpot_slo

    # Determine request source parameters
    if request_source == 'mooncake':
        request_param = args.trace_path
    elif request_source == 'lognormal':
        request_param = args.lognormal_mean
    elif request_source == 'zipfian':
        request_param = args.zipf_s
    else:
        raise RuntimeError

    with open(args.prefill_config) as f:
        prefill_config = json.load(f)
    with open(args.decode_config) as f:
        decode_config = json.load(f)

    model_id = args.model_id

    if args.resume:
        with open(args.checkpoint_path, 'rb') as f:
            all_results, raw_df_map = pickle.load(f)
    else:
        # Generate all parameter combinations
        param_combinations = list(itertools.product(
            lambda_arrivals, prefill_globals, prefill_locals, decode_globals,
            decode_locals, ttft_slos, tpot_slos
        ))
        total_combinations = len(param_combinations)

        print(f"{total_combinations} simulations in total.")

        # Prepare the partial function with fixed arguments
        run_sim_partial = partial(
            run_single_simulation,
            request_count=request_count,
            request_source=request_source,
            request_param=request_param,
            prefill_config=prefill_config,
            decode_config=decode_config,
            seed=seed,
            model_id=model_id,
            sample_queue_states=args.sample_queue_states,
            sample_memory_usage=args.sample_memory_usage,
            sample_rate=args.sample_rate,
            queue_states_output=args.queue_states_output,
            memory_usage_output=args.memory_usage_output
        )

        # Run simulations in parallel
        with mp.Pool(processes=args.n_jobs) as pool:
            results = pool.map(run_sim_partial, param_combinations)

        # Process results
        all_results: List[pd.DataFrame] = []
        raw_df_map: Dict[str, pd.DataFrame] = {}

        for i, (df, stats_df, params) in enumerate(results, 1):
            lambda_arrival, prefill_global, prefill_local, decode_global, decode_local, ttft_slo, tpot_slo = params

            print(f"\n[{i}/{total_combinations}] Results for:")
            print(
                f"  lambda_arrival={lambda_arrival}, prefill_global={prefill_global}, prefill_local={prefill_local}")
            print(
                f"  decode_global={decode_global}, decode_local={decode_local}, ttft_slo={ttft_slo}, tpot_slo={tpot_slo}")

            # Print statistics to console
            print(
                f"[{prefill_global:<12} | {prefill_local:<8}] "
                f"p25={stats_df['ttft_p25'].iloc[0]:.4f}s   "
                f"p50={stats_df['ttft_p50'].iloc[0]:.4f}s   "
                f"p75={stats_df['ttft_p75'].iloc[0]:.4f}s   "
                f"p90={stats_df['ttft_p90'].iloc[0]:.4f}s   "
                f"p95={stats_df['ttft_p95'].iloc[0]:.4f}s   "
                f"p99={stats_df['ttft_p99'].iloc[0]:.4f}s   "
                f"slo attainment={stats_df['slo_attainment_pct'].iloc[0]:.4f}%   "
                f"mean={stats_df['ttft_mean'].iloc[0]:.4f}s"
            )

            raw_df_map[stats_df['uuid'].iloc[0]] = df
            all_results.append(stats_df)

        # Serialize and recover later to reduce computation
        with open(args.checkpoint_path, 'wb') as f:
            pickle.dump((all_results, raw_df_map), f)

    # Save all results to CSV if requested
    if args.csv_output and all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(args.csv_output, index=False)
        print(
            f"\nAll {len(all_results)} simulation results saved to {args.csv_output}")

    # Visualize the results
    visualizer = TTFTVisualizer(all_results, raw_df_map)
    visualizer.plot_ttft_vs_length("figures")
