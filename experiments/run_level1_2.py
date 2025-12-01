# experiments/run_level1_2.py
#
# Round 2 experiment: add semantic drift across u, richer per-position entropy,
# and representative completions, without overwriting round 1 outputs.

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu

from config import ExperimentConfig
from data_utils import sample_wikipedia_prompts
from model_utils import load_model, generate_with_fixed_u
from metrics import aggregate_metrics, embed_texts
from plots import plot_position_entropy_heatmap


def pairwise_cosine_matrix(embeddings):
    """embeddings: tensor (num_u, hidden)."""
    n = embeddings.shape[0]
    sims = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sims[i, j] = F.cosine_similarity(embeddings[i], embeddings[j], dim=0)
    return sims


def pairwise_bleu_matrix(texts):
    """texts: list of len num_u."""
    n = len(texts)
    mat = np.zeros((n, n), dtype=float)
    tokenized = [t.split() for t in texts]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
            else:
                # symmetrize by averaging BLEU(i|j) and BLEU(j|i)
                b1 = sentence_bleu([tokenized[j]], tokenized[i])
                b2 = sentence_bleu([tokenized[i]], tokenized[j])
                mat[i, j] = 0.5 * (b1 + b2)
    return mat


def save_heatmap(matrix, u_values, title, output_path, vmin=None, vmax=None):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
               extent=[0, len(u_values), 0, len(u_values)])
    plt.colorbar()
    plt.xticks(ticks=range(len(u_values)), labels=[str(u) for u in u_values], rotation=90)
    plt.yticks(ticks=range(len(u_values)), labels=[str(u) for u in u_values])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_level1_round2():
    # keep outputs separate from round 1
    cfg = ExperimentConfig(
        num_prompts=50,
        num_samples_per_u=20,
        results_csv="results/level1_round2_metrics.csv",
        entropy_npy="results/position_entropy_round2.npy",
        logs_dir="logs_round2"
    )
    plots_dir = "plots_round2"
    os.makedirs(plots_dir, exist_ok=True)

    model, tokenizer, device = load_model(cfg.model_name)
    embed_model = model.transformer

    prompts = sample_wikipedia_prompts(num_prompts=cfg.num_prompts)
    print(f"Sampled {len(prompts)} Wikipedia prompts.")

    all_results = []
    pos_entropy_dict = {}
    csv_has_header = os.path.exists(cfg.results_csv) and os.path.getsize(cfg.results_csv) > 0

    rep_texts = {}        # prompt_idx -> list of rep texts per u
    rep_embeddings = {}   # prompt_idx -> tensor (num_u, hidden)

    for p_idx, prompt in enumerate(prompts):
        prompt_slug = f"prompt{p_idx}"
        log_file = os.path.join(cfg.logs_dir, f"{prompt_slug}.log")
        os.makedirs(cfg.logs_dir, exist_ok=True)
        with open(log_file, "w") as f:
            f.write(f"=== Prompt {p_idx} ===\n")
            f.write(prompt + "\n\n")

        print(f"\n=== Prompt {p_idx+1}/{len(prompts)} ===")
        print(prompt[:200] + ("..." if len(prompt) > 200 else ""))

        per_u_texts = []
        per_u_embeds = []

        for u in cfg.u_values:
            print(f"  -> u={u}")
            with open(log_file, "a") as f:
                f.write(f"\n=== u={u} ===\n")

            samples = []
            for run_idx in range(cfg.num_samples_per_u):
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

            pd.DataFrame([metrics_row]).to_csv(
                cfg.results_csv,
                mode="a",
                header=not csv_has_header,
                index=False
            )
            csv_has_header = True

            keys = list(pos_entropy_dict.keys())
            values = np.stack([pos_entropy_dict[k] for k in keys], axis=0)
            np.save(cfg.entropy_npy, {"keys": keys, "values": values}, allow_pickle=True)

            # representative sample for semantic drift
            rep_text = samples[0]
            per_u_texts.append(rep_text)
            rep_embed = embed_texts(embed_model, tokenizer, [rep_text], device=device)[0].detach().cpu()
            per_u_embeds.append(rep_embed)

        rep_texts[p_idx] = per_u_texts
        rep_embeddings[p_idx] = torch.stack(per_u_embeds, dim=0)

    # final saves
    df = pd.DataFrame(all_results)
    df.to_csv(cfg.results_csv, index=False)
    print(f"\nSaved metrics to {cfg.results_csv}")

    keys = list(pos_entropy_dict.keys())
    values = np.stack([pos_entropy_dict[k] for k in keys], axis=0)
    np.save(cfg.entropy_npy, {"keys": keys, "values": values}, allow_pickle=True)
    print(f"Saved position entropies to {cfg.entropy_npy}")

    # position entropy heatmap (averaged over prompts)
    plot_position_entropy_heatmap(
        pos_entropy_dict,
        max_len=cfg.max_len,
        output_path=os.path.join(plots_dir, "position_entropy_heatmap.png")
    )

    # semantic drift: average pairwise cosine and BLEU over prompts
    num_u = len(cfg.u_values)
    cos_sum = np.zeros((num_u, num_u), dtype=float)
    cos_sq = np.zeros((num_u, num_u), dtype=float)
    bleu_sum = np.zeros((num_u, num_u), dtype=float)
    bleu_sq = np.zeros((num_u, num_u), dtype=float)

    for p_idx in rep_embeddings:
        emb = rep_embeddings[p_idx]
        texts = rep_texts[p_idx]

        cos_mat = pairwise_cosine_matrix(emb).numpy()
        bleu_mat = pairwise_bleu_matrix(texts)

        cos_sum += cos_mat
        cos_sq += cos_mat ** 2
        bleu_sum += bleu_mat
        bleu_sq += bleu_mat ** 2

    n_prompts = len(rep_embeddings)
    cos_mean = cos_sum / n_prompts
    cos_std = np.sqrt(cos_sq / n_prompts - cos_mean ** 2)
    bleu_mean = bleu_sum / n_prompts
    bleu_std = np.sqrt(bleu_sq / n_prompts - bleu_mean ** 2)

    np.save("results/semantic_cosine_round2.npy", {"mean": cos_mean, "std": cos_std, "u_values": cfg.u_values})
    np.save("results/semantic_bleu_round2.npy", {"mean": bleu_mean, "std": bleu_std, "u_values": cfg.u_values})

    save_heatmap(
        cos_mean,
        cfg.u_values,
        "Mean cosine similarity across u",
        os.path.join(plots_dir, "semantic_cosine_heatmap.png"),
        vmin=0.0,
        vmax=1.0,
    )
    save_heatmap(
        bleu_mean,
        cfg.u_values,
        "Mean BLEU across u",
        os.path.join(plots_dir, "semantic_bleu_heatmap.png"),
        vmin=0.0,
        vmax=1.0,
    )

    # Representative completions for readability
    rep_u_values = [u for u in [0.0, 0.3, 0.6, 0.9] if u in cfg.u_values]
    reps = []
    for p_idx in sorted(rep_texts.keys())[:4]:
        text_map = dict(zip(cfg.u_values, rep_texts[p_idx]))
        for u in rep_u_values:
            reps.append({
                "prompt_idx": p_idx,
                "u": u,
                "text": text_map.get(u, "")
            })
    pd.DataFrame(reps).to_csv("results/representative_completions_round2.csv", index=False)

    print("Saved semantic drift matrices and representative completions.")
    print("Done with round 2.")


if __name__ == "__main__":
    run_level1_round2()
