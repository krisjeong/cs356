# experiments/config.py

from dataclasses import dataclass
import numpy as np
import os

@dataclass
class ExperimentConfig:
    model_name: str = "gpt2"
    max_len: int = 80          # max generation length
    num_samples_per_u: int = 1  # number of runs per (prompt, u)
    u_values: list = None
    num_prompts: int = 50     # number of Wikipedia prompts to sample

    logs_dir: str = "logs"
    results_csv: str = "results/level1_results.csv"
    entropy_npy: str = "results/position_entropy.npy"

    def __post_init__(self):
        if self.u_values is None:
            # e.g. 21 values between 0 and 1 inclusive
            self.u_values = [float(round(x, 2)) for x in np.linspace(0.0, 1.0, 21)]
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
