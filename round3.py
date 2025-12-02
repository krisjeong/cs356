import os
import random
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu


# =========================
# Config
# =========================

@dataclass
class Round3Config:
    model_name: str = "gpt2"
    max_len: int = 80
    num_prompts: int = 50
    seed: int = 42

    # scalar grid
    u_values: list = None

    # dirs
    logs_dir: str = "logs_round3"
    results_dir: str = "results_round3"
    plots_dir: str = "plots_round3"

    # sentiment attack
    sentiment_beta: int = 4          # every beta tokens use adversarial u
    sentiment_top_k: int = 50
    negative_band: tuple = (0.41, 0.86)
    use_sentiment_prompts: bool = False  # optionally reuse sentiment prompts for scalar geometry

    def __post_init__(self):
        if self.u_values is None:
            self.u_values = [float(round(x, 2)) for x in np.linspace(0.0, 1.0, 21)]

        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)


# =========================
# Utilities
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(model_name="gpt2", device=None):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def sample_wikipedia_prompts(
    num_prompts=50,
    min_chars=60,
    max_chars=200,
    seed=42,
):
    random.seed(seed)
    stream = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        streaming=True
    )["train"]

    prompts = []
    for article in stream:
        if len(prompts) >= num_prompts:
            break

        text = article["text"]
        text = text.strip().replace("\n", " ")
        if len(text) < min_chars:
            continue

        start = random.randint(0, max(0, len(text) - max_chars))
        end = start + random.randint(min_chars, max_chars)
        snippet = text[start:end].strip()
        if len(snippet.split()) < 5:
            continue
        prompts.append(snippet)

    return prompts


# ---------- Generators ----------

def generate_with_fixed_u(model, tokenizer, prompt, u, max_len, device):
    """
    Deterministic fixed-u sampling used for scalar geometry.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1).squeeze(0)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=0)
            idx = torch.searchsorted(cdf, torch.tensor(u, device=device))
            idx = torch.clamp(idx, max=cdf.numel() - 1)
            chosen_token_id = sorted_indices[idx].view(1, 1)
            generated = torch.cat([generated, chosen_token_id], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def generate_topk_normal(model, tokenizer, prompt, max_len, device, top_k=50):
    """
    Baseline stochastic top-k sampler.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1).squeeze(0)

            topk_probs, topk_indices = torch.topk(probs, k=top_k)
            topk_probs = topk_probs / topk_probs.sum()
            cdf = torch.cumsum(topk_probs, dim=0)

            u = np.random.rand()
            idx = torch.searchsorted(cdf, torch.tensor(u, device=device))
            chosen_token_id = topk_indices[idx].view(1, 1)
            generated = torch.cat([generated, chosen_token_id], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def generate_topk_adversarial_schedule(
    model,
    tokenizer,
    prompt,
    max_len,
    device,
    beta,
    negative_band=(0.41, 0.86),
    top_k=50,
):
    """
    Every beta-th token, sample u from negative_band; elsewhere sample u~Uniform[0,1].
    Always sample from top-k tokens.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for t in range(max_len):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1).squeeze(0)

            topk_probs, topk_indices = torch.topk(probs, k=top_k)
            topk_probs = topk_probs / topk_probs.sum()
            cdf = torch.cumsum(topk_probs, dim=0)

            if t % beta == 0:
                u = np.random.uniform(*negative_band)
            else:
                u = np.random.rand()

            idx = torch.searchsorted(cdf, torch.tensor(u, device=device))
            chosen_token_id = topk_indices[idx].view(1, 1)
            generated = torch.cat([generated, chosen_token_id], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


# ---------- Metrics ----------

def token_stats(text, tokenizer):
    tokens = tokenizer.encode(text)
    L = len(tokens)
    unique_ratio = len(set(tokens)) / L if L > 0 else 0.0

    counts = Counter(tokens)
    freqs = np.array(list(counts.values()), dtype=float)
    if L > 0:
        freqs /= L
        entropy = -(freqs * np.log(freqs)).sum()
    else:
        entropy = 0.0

    repetitions = (
        sum(tokens[i] == tokens[i - 1] for i in range(1, L)) / L
        if L > 1 else 0.0
    )

    return {
        "length": L,
        "unique_ratio": unique_ratio,
        "entropy": float(entropy),
        "repetition": float(repetitions),
    }


def position_entropy_across_prompts(completions_by_u, tokenizer, max_len):
    """
    completions_by_u: dict[u] -> list[text] (one per prompt)
    returns: sorted u_values, entropy_matrix (num_u, max_len)
    """
    u_values = sorted(completions_by_u.keys())
    num_u = len(u_values)
    entropy_matrix = np.zeros((num_u, max_len), dtype=float)

    for ui, u in enumerate(u_values):
        texts = completions_by_u[u]
        encoded = [tokenizer.encode(t) for t in texts]

        for t in range(max_len):
            toks = []
            for seq in encoded:
                if t < len(seq):
                    toks.append(seq[t])
            if not toks:
                entropy_matrix[ui, t] = 0.0
                continue

            counts = Counter(toks)
            total = sum(counts.values())
            freqs = np.array(list(counts.values()), dtype=float) / total
            ent = -(freqs * np.log(freqs)).sum()
            entropy_matrix[ui, t] = float(ent)

    return u_values, entropy_matrix


def encode_embedding(embed_model, tokenizer, text, device):
    with torch.no_grad():
        ids = tokenizer.encode(text, return_tensors="pt").to(device)
        outputs = embed_model(ids)
        hidden = outputs.last_hidden_state.mean(dim=1)
    return hidden.squeeze(0)


def cosine_matrix_across_u(embeddings_by_u, u_values):
    num_u = len(u_values)
    cosine_mat = np.zeros((num_u, num_u), dtype=float)

    for i, ui in enumerate(u_values):
        for j, uj in enumerate(u_values):
            sims = []
            emb_i_list = embeddings_by_u[ui]
            emb_j_list = embeddings_by_u[uj]
            assert len(emb_i_list) == len(emb_j_list)
            for e_i, e_j in zip(emb_i_list, emb_j_list):
                sims.append(F.cosine_similarity(e_i, e_j, dim=0).item())
            cosine_mat[i, j] = float(np.mean(sims)) if sims else 0.0
    return cosine_mat


def count_negations(text):
    lower = text.lower()
    tokens = lower.split()
    negs = ["no", "not", "never"]
    cnt = sum(tok in negs or tok.endswith("n't") for tok in tokens)
    return cnt


# =========================
# Plotting
# =========================

def plot_position_entropy_heatmap(u_values, entropy_matrix, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    max_len = entropy_matrix.shape[1]
    plt.figure(figsize=(10, 4))
    plt.imshow(
        entropy_matrix,
        aspect="auto",
        origin="lower",
        extent=[0, max_len, 0, len(u_values)]
    )
    plt.colorbar(label="Token entropy")
    plt.yticks(range(len(u_values)), [str(u) for u in u_values])
    plt.xlabel("Position (t)")
    plt.ylabel("u")
    plt.title("Per-position token entropy vs. u (across prompts)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_cosine_heatmap(u_values, cosine_mat, output_path, title):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cosine_mat, origin="lower", aspect="equal", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Mean cosine similarity")
    ticks = range(len(u_values))
    labels = [str(u) for u in u_values]
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    plt.xlabel("u")
    plt.ylabel("u")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_sentiment_boxplot(baseline_scores, adv_scores, output_path):
    plt.figure(figsize=(5, 4))
    plt.boxplot([baseline_scores, adv_scores], labels=["Baseline", "Adversarial"])
    plt.ylabel("Sentiment score (positive prob - negative prob)")
    plt.title("Sentiment shift under sparse negative-band scalars")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_bleu_hist(bleu_scores, output_path):
    plt.figure(figsize=(5, 4))
    plt.hist(bleu_scores, bins=10, alpha=0.8)
    plt.xlabel("BLEU(baseline, adversarial)")
    plt.ylabel("Count")
    plt.title("Lexical similarity between baseline and adversarial outputs")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_negation_bar(baseline_counts, adv_counts, output_path):
    plt.figure(figsize=(5, 4))
    x = np.arange(2)
    means = [np.mean(baseline_counts), np.mean(adv_counts)]
    stds = [np.std(baseline_counts), np.std(adv_counts)]
    plt.bar(x, means, yerr=stds, tick_label=["Baseline", "Adversarial"], alpha=0.8)
    plt.ylabel("Avg # of negation tokens")
    plt.title("Negation frequency under sparse negative-band scalars")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =========================
# Round 3: scalar geometry
# =========================

def run_round3_scalar_geometry(cfg, model, tokenizer, device):
    print("=== Round 3: scalar geometry (position entropy + embedding drift) ===")

    if getattr(cfg, "use_sentiment_prompts", False):
        prompts = get_sentiment_prompts()[: cfg.num_prompts]
        print(f"Using sentiment prompts (n={len(prompts)}) instead of Wikipedia sampling.")
    else:
        prompts = sample_wikipedia_prompts(
            num_prompts=cfg.num_prompts,
            min_chars=60,
            max_chars=200,
            seed=cfg.seed,
        )
    print(f"Sampled {len(prompts)} prompts.")

    completions_by_u = {u: [] for u in cfg.u_values}
    embeddings_by_u = {u: [] for u in cfg.u_values}
    rows = []

    embed_model = model.transformer

    for p_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        prompt_slug = f"prompt_{p_idx:03d}"
        log_path = os.path.join(cfg.logs_dir, f"{prompt_slug}.log")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Prompt {p_idx} ===\n")
            f.write(prompt + "\n\n")

        for u in cfg.u_values:
            text = generate_with_fixed_u(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                u=u,
                max_len=cfg.max_len,
                device=device,
            )

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== u = {u} ===\n")
                f.write(text + "\n")

            completions_by_u[u].append(text)
            emb = encode_embedding(embed_model, tokenizer, text, device=device)
            embeddings_by_u[u].append(emb.cpu())

            stats = token_stats(text, tokenizer)
            row = {
                "prompt_idx": p_idx,
                "prompt": prompt,
                "u": u,
                "completion": text,
                **stats,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    completions_csv = os.path.join(cfg.results_dir, "completions_round3.csv")
    df.to_csv(completions_csv, index=False)
    print(f"Saved completions + stats to {completions_csv}")

    # Position-wise entropy across prompts
    u_values, entropy_matrix = position_entropy_across_prompts(
        completions_by_u, tokenizer, max_len=cfg.max_len
    )
    pos_entropy_npy = os.path.join(cfg.results_dir, "position_entropy_round3.npy")
    np.save(pos_entropy_npy, {"u_values": np.array(u_values),
                              "entropy_matrix": entropy_matrix},
            allow_pickle=True)
    print(f"Saved position entropy to {pos_entropy_npy}")

    entropy_plot_path = os.path.join(cfg.plots_dir, "position_entropy_heatmap_round3.png")
    plot_position_entropy_heatmap(u_values, entropy_matrix, entropy_plot_path)
    print(f"Saved position entropy heatmap to {entropy_plot_path}")

    # Embedding cosine similarity across u
    cosine_mat = cosine_matrix_across_u(embeddings_by_u, u_values)
    cosine_npy = os.path.join(cfg.results_dir, "embedding_cosine_round3.npy")
    np.save(cosine_npy, {"u_values": np.array(u_values),
                         "cosine_matrix": cosine_mat},
            allow_pickle=True)
    print(f"Saved embedding cosine matrix to {cosine_npy}")

    cosine_plot_path = os.path.join(cfg.plots_dir, "embedding_cosine_heatmap_round3.png")
    plot_cosine_heatmap(
        u_values,
        cosine_mat,
        cosine_plot_path,
        title="Mean cosine similarity across u (Round 3)",
    )
    print(f"Saved embedding cosine heatmap to {cosine_plot_path}")


# =========================
# Sentiment attack experiment
# =========================

def get_sentiment_prompts():
    return [
        "The product was",
        "Overall, my experience with this service was",
        "I would recommend this to my friend because",
        "This policy is",
        "The movie made me feel",
        "The food at the restaurant was",
        "I think this decision is",
        "The vacation turned out to be",
        "The teacher was",
        "My day today was",
    ]


def run_sentiment_attack(cfg, model, tokenizer, device):
    print("=== Sentiment attack experiment (sparse negative-band scalars) ===")

    prompts = get_sentiment_prompts()
    sentiment_analyzer = pipeline("sentiment-analysis", device=0 if device == "cuda" else -1)

    rows = []
    baseline_scores = []
    adv_scores = []
    bleu_scores = []
    baseline_neg_counts = []
    adv_neg_counts = []

    for p_idx, prompt in enumerate(tqdm(prompts, desc="Sentiment prompts")):
        log_path = os.path.join(cfg.logs_dir, f"sentiment_prompt_{p_idx:03d}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Sentiment Prompt {p_idx} ===\n")
            f.write(prompt + "\n\n")

        # Baseline
        baseline = generate_topk_normal(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_len=cfg.max_len,
            device=device,
            top_k=cfg.sentiment_top_k,
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("=== BASELINE ===\n")
            f.write(baseline + "\n\n")

        # Adversarial
        adversarial = generate_topk_adversarial_schedule(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_len=cfg.max_len,
            device=device,
            beta=cfg.sentiment_beta,
            negative_band=cfg.negative_band,
            top_k=cfg.sentiment_top_k,
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("=== ADVERSARIAL ===\n")
            f.write(adversarial + "\n")

        # Sentiment scores
        base_res = sentiment_analyzer(baseline)[0]
        adv_res = sentiment_analyzer(adversarial)[0]

        def signed_score(res):
            score = res["score"]
            return score if res["label"].upper().startswith("POS") else -score

        base_score = signed_score(base_res)
        adv_score = signed_score(adv_res)

        # BLEU between baseline and adversarial
        bleu = sentence_bleu(
            [baseline.split()],
            adversarial.split()
        )

        # Negation counts
        base_neg = count_negations(baseline)
        adv_neg = count_negations(adversarial)

        baseline_scores.append(base_score)
        adv_scores.append(adv_score)
        bleu_scores.append(bleu)
        baseline_neg_counts.append(base_neg)
        adv_neg_counts.append(adv_neg)

        rows.append({
            "prompt_idx": p_idx,
            "prompt": prompt,
            "baseline": baseline,
            "adversarial": adversarial,
            "baseline_sentiment": base_score,
            "adversarial_sentiment": adv_score,
            "bleu": bleu,
            "baseline_negations": base_neg,
            "adversarial_negations": adv_neg,
        })

    # Save CSV
    sentiment_csv = os.path.join(cfg.results_dir, "sentiment_attack_round3.csv")
    pd.DataFrame(rows).to_csv(sentiment_csv, index=False)
    print(f"Saved sentiment attack results to {sentiment_csv}")

    # Plots
    sentiment_box_path = os.path.join(cfg.plots_dir, "sentiment_boxplot_round3.png")
    plot_sentiment_boxplot(baseline_scores, adv_scores, sentiment_box_path)
    print(f"Saved sentiment boxplot to {sentiment_box_path}")

    bleu_hist_path = os.path.join(cfg.plots_dir, "sentiment_bleu_hist_round3.png")
    plot_bleu_hist(bleu_scores, bleu_hist_path)
    print(f"Saved BLEU histogram to {bleu_hist_path}")

    neg_bar_path = os.path.join(cfg.plots_dir, "sentiment_negation_bar_round3.png")
    plot_negation_bar(baseline_neg_counts, adv_neg_counts, neg_bar_path)
    print(f"Saved negation barplot to {neg_bar_path}")


# =========================
# Main
# =========================

def main():
    cfg = Round3Config()
    set_seed(cfg.seed)

    model, tokenizer, device = load_model(cfg.model_name)
    print(f"Using device: {device}")

    # Round 3 scalar geometry (position entropy + embedding drift)
    run_round3_scalar_geometry(cfg, model, tokenizer, device)

    # Sentiment attack experiment (sparse negative-band scalars)
    run_sentiment_attack(cfg, model, tokenizer, device)

    print("All Round 3 experiments completed.")


if __name__ == "__main__":
    main()
