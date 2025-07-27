import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from pathlib import Path

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TTFTVisualizer:
    def __init__(self, all_results: List[pd.DataFrame], raw_df_map: Dict[str, pd.DataFrame]):
        """Initialize the TTFT visualizer with results data."""
        self.all_results = all_results
        self.raw_df_map = raw_df_map
        self.setup_plotting_style()

    def setup_plotting_style(self):
        """Setup matplotlib and seaborn styling."""
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10

    def plot_ttft_vs_length(self, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        combined_results = pd.concat(self.all_results, ignore_index=True)
        grouped_results = combined_results.groupby('lambda_arrival')
        
        for lambda_arrival, results in grouped_results:
            for uuid in results['uuid'].unique():
                df = self.raw_df_map[uuid]
                if len(df) > 5000:
                    df_sampled = df.sample(n=5000, random_state=42)
                else:
                    df_sampled = df
                plt.scatter(df_sampled['input_length'], df_sampled['ttft'], alpha=0.5, label=uuid)
            plt.title(f"TTFT vs Length for Lambda_arrival={lambda_arrival}")
            plt.xlabel("Input Length")
            plt.ylabel("TTFT")
            plt.legend()
            plt.show()
            plt.savefig(f"{output_dir}/{lambda_arrival}.png")
            plt.close()
