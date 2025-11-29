# experiments/run_level1.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import ExperimentConfig
from data_utils import sample_wikipedia_prompts
from model_utils import load_model, generate_with_fixed_u
from metrics import aggregate_metrics
from plots import plot_metric_vs_u, plot_position_entropy_heatmap


def run_level1_experiments():
    cfg = ExperimentConfig()

    # === Model & prompts ===
    model, tokenizer, device = load_model(cfg.model_name)
    # Use model itself for embeddings (could also be a separate encoder)
    embed_model = model.transformer

    prompts = sample_wikipedia_prompts(num_prompts=cfg.num_prompts)
    print(f"Sampled {len(prompts)} Wikipedia prompts.")

    all_results = []
    pos_entropy_dict = {}  # (prompt_idx, u) -> np.array(max_len,)
    # Track whether we've already written a header to the CSV
    csv_has_header = os.path.exists(cfg.results_csv) and os.path.getsize(cfg.results_csv) > 0

    for p_idx, prompt in enumerate(prompts):
        prompt_slug = f"prompt{p_idx}"
        log_file = os.path.join(cfg.logs_dir, f"{prompt_slug}.log")

        with open(log_file, "w") as f:
            f.write(f"=== Prompt {p_idx} ===\n")
            f.write(prompt + "\n\n")

        print(f"\n=== Prompt {p_idx+1}/{len(prompts)} ===")
        print(prompt[:200] + ("..." if len(prompt) > 200 else ""))

        for u in cfg.u_values:
            print(f"  -> u={u}")
            with open(log_file, "a") as f:
                f.write(f"\n=== u={u} ===\n")

            samples = []
            for run_idx in range(cfg.num_samples_per_u):
                # log every run so progress is persisted frequently
                msg = f"    [u={u}] Run {run_idx+1}/{cfg.num_samples_per_u}"
                print(msg)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")

                text = generate_with_fixed_u(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    u=u,
                    max_len=cfg.max_len,
                    device=device
                )
                samples.append(text)

            # Compute metrics & position entropy
            metrics, pos_ent = aggregate_metrics(
                samples,
                tokenizer,
                embed_model,
                device=device,
                max_len=cfg.max_len
            )
            metrics_row = {
                "prompt_idx": p_idx,
                "prompt": prompt,
                "u": u,
                **metrics,
            }
            all_results.append(metrics_row)
            pos_entropy_dict[(p_idx, u)] = pos_ent

            # Write metrics incrementally to CSV (append mode)
            pd.DataFrame([metrics_row]).to_csv(
                cfg.results_csv,
                mode="a",
                header=not csv_has_header,
                index=False
            )
            csv_has_header = True

            # Save entropy snapshot frequently to avoid losing progress
            keys = list(pos_entropy_dict.keys())
            values = np.stack([pos_entropy_dict[k] for k in keys], axis=0)
            np.save(cfg.entropy_npy, {"keys": keys, "values": values}, allow_pickle=True)

    # === Save metrics dataframe ===
    df = pd.DataFrame(all_results)
    df.to_csv(cfg.results_csv, index=False)
    print(f"\nSaved metrics to {cfg.results_csv}")

    # === Save per-position entropies ===
    # We’ll save as a dict serialized via numpy
    keys = list(pos_entropy_dict.keys())
    values = np.stack([pos_entropy_dict[k] for k in keys], axis=0)
    np.save(cfg.entropy_npy, {"keys": keys, "values": values}, allow_pickle=True)
    print(f"Saved position entropies to {cfg.entropy_npy}")

    # === Generate plots ===
    metrics_to_plot = [
        "length",
        "unique_ratio",
        "entropy",
        "repetition",
        "self_bleu",
        "embedding_sim",
    ]
    for metric in metrics_to_plot:
        print(f"Plotting {metric} vs u ...")
        plot_metric_vs_u(df, metric, output_dir="plots")

    print("Plotting position entropy heatmap ...")
    plot_position_entropy_heatmap(
        pos_entropy_dict,
        max_len=cfg.max_len,
        output_path="plots/position_entropy_heatmap.png"
    )

    print("Done.")


if __name__ == "__main__":
    run_level1_experiments()
