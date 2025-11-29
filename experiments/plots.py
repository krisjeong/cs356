# experiments/plots.py

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_metric_vs_u(df, metric, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6,4))

    grouped = df.groupby("u")[metric].agg(["mean", "std"]).reset_index()
    us = grouped["u"].values
    means = grouped["mean"].values
    stds = grouped["std"].values

    plt.plot(us, means, marker="o")
    plt.fill_between(us, means-stds, means+stds, alpha=0.2)

    plt.title(f"{metric} vs. u (mean ± std across prompts)")
    plt.xlabel("u")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_vs_u.png"))
    plt.close()


def plot_position_entropy_heatmap(pos_entropy_dict, max_len, output_path="plots/position_entropy_heatmap.png"):
    """
    pos_entropy_dict: dict[(prompt_idx, u)] -> np.array (max_len,)
    We average over prompts to get an entropy matrix of shape (len(u_values), max_len).
    """
    # infer u_values
    u_values = sorted({key[1] for key in pos_entropy_dict.keys()})
    u_to_idx = {u: i for i, u in enumerate(u_values)}
    num_us = len(u_values)

    # build matrix: (num_us, max_len): averaged over prompts
    accum = np.zeros((num_us, max_len), dtype=float)
    counts = np.zeros(num_us, dtype=int)

    for (prompt_idx, u), ent_vec in pos_entropy_dict.items():
        ui = u_to_idx[u]
        accum[ui] += ent_vec
        counts[ui] += 1

    for i in range(num_us):
        if counts[i] > 0:
            accum[i] /= counts[i]

    plt.figure(figsize=(10, 4))
    plt.imshow(accum, aspect="auto", origin="lower",
               extent=[0, max_len, 0, num_us])

    plt.colorbar(label="Token entropy")
    plt.yticks(range(num_us), [str(u) for u in u_values])
    plt.xlabel("Position (t)")
    plt.ylabel("u")
    plt.title("Per-position token entropy vs. u (averaged over prompts)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
