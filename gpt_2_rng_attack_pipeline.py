# GPT-2 PRNG Manipulation Research Pipeline
# Clean, modular, research-ready notebook-style Python script.

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
import torch.nn.functional as F
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Model + Tokenizer Setup
# ---------------------------------------------------------------
def load_model(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

# ---------------------------------------------------------------
# Adversarial Sampler (Level 1)
# ---------------------------------------------------------------
# def generate_with_u(model, tokenizer, prompt, u, max_len=80):
#     input_ids = tokenizer.encode(prompt, return_tensors='pt')
#     generated = input_ids.clone()

#     for _ in range(max_len):
#         outputs = model(generated)
#         logits = outputs.logits[:, -1, :]
#         probs = torch.softmax(logits, dim=-1).squeeze()

#         cdf = torch.cumsum(probs, dim=0)
#         k = torch.searchsorted(cdf, torch.tensor(u))

#         # FIX: match batch dimension
#         k = k.view(1, 1)  # shape becomes (1,1)

#         generated = torch.cat([generated, k], dim=1)

#     return tokenizer.decode(generated[0])

def generate_with_u_debug(
    model,
    tokenizer,
    prompt,
    u,
    max_len=50,
    log_path=None,
    print_console=True
):
    def log(msg):
        if print_console:
            print(msg)
        if log_path:
            with open(log_path, "a") as f:
                f.write(msg + "\n")

    if log_path:
        with open(log_path, "w") as f:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"=== LOG START ({ts}) ===\n")

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated = input_ids.clone()

    log(f"\n=== GENERATING WITH u = {u} ===")
    log(f"Prompt: {prompt}\n")

    for step in range(max_len):
        # forward pass
        outputs = model(generated)
        logits = outputs.logits[:, -1, :]

        # softmax
        probs = torch.softmax(logits, dim=-1).squeeze()

        # sort probabilities DESCENDING
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # compute CDF over sorted probs
        cdf = torch.cumsum(sorted_probs, dim=0)

        # find first index where CDF >= u
        idx = torch.searchsorted(cdf, torch.tensor(u))

        # map back to the actual token ID
        chosen_token_id = sorted_indices[idx].unsqueeze(0).unsqueeze(0)

        # decode token
        token_str = tokenizer.decode(chosen_token_id[0])
        max_prob = float(sorted_probs[0])

        # update sequence
        new_generated = torch.cat([generated, chosen_token_id], dim=1)
        partial_text = tokenizer.decode(new_generated[0])

        # log step
        log(f"Step {step+1}")
        log(f"  u: {u}")
        log(f"  chosen token id: {chosen_token_id.item()} ({repr(token_str)})")
        log(f"  highest prob token: {max_prob:.5f}")
        log(f"  sorted index chosen: {idx.item()}")
        log(f"  Generated so far: {partial_text}")
        log("-" * 50)

        # update state
        generated = new_generated

    final_text = tokenizer.decode(generated[0])
    log("\n=== FINAL OUTPUT ===")
    log(final_text)
    log("=" * 50)

    return final_text




# ---------------------------------------------------------------
# Batch Generation (20 deterministic samples per u)
# ---------------------------------------------------------------
def batch_generate(model, tokenizer, prompt, u, n=20, max_len=80):
    # return [generate_with_u(model, tokenizer, prompt, u, max_len) for _ in range(n)]
    return [generate_with_u_debug(model, tokenizer, prompt, u, max_len, log_path="generation_log_u0.1.txt", print_console=True) for _ in range(n)]


# ---------------------------------------------------------------
# Token-Level Metrics
# ---------------------------------------------------------------
def token_stats(text, tokenizer):
    tokens = tokenizer.encode(text)
    L = len(tokens)
    unique_ratio = len(set(tokens)) / L if L > 0 else 0

    counts = Counter(tokens)
    freqs = np.array(list(counts.values())) / L if L > 0 else np.array([1.0])
    entropy = -(freqs * np.log(freqs)).sum()

    repetitions = sum(tokens[i] == tokens[i-1] for i in range(1, L)) / L if L > 1 else 0

    return {
        "length": L,
        "unique_ratio": unique_ratio,
        "entropy": entropy,
        "repetition": repetitions
    }

# ---------------------------------------------------------------
# Sequence-Level Metrics
# ---------------------------------------------------------------
def self_bleu(samples):
    scores = []
    for i, s in enumerate(samples):
        refs = [samples[j].split() for j in range(len(samples)) if j != i]
        scores.append(sentence_bleu(refs, s.split()))
    return float(np.mean(scores))


def embed_texts(model, tokenizer, samples):
    embeddings = []
    for s in samples:
        ids = tokenizer.encode(s, return_tensors='pt')
        with torch.no_grad():
            hidden = model.transformer(ids).last_hidden_state.mean(dim=1)
        embeddings.append(hidden)
    return torch.stack(embeddings).squeeze()


def avg_cosine_similarity(embeddings):
    sims = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sims.append(F.cosine_similarity(embeddings[i], embeddings[j], dim=0).item())
    return float(np.mean(sims))

def plot_metric_vs_u(df, metric, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8,5))
    for prompt in df["prompt"].unique():
        subset = df[df["prompt"] == prompt]
        plt.plot(subset["u"], subset[metric], marker="o", label=prompt[:35] + "...")

    plt.title(f"{metric} vs. u-value")
    plt.xlabel("u-value (quantile sampling threshold)")
    plt.ylabel(metric)
    plt.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}_vs_u.png")
    plt.close()

# ---------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------
# def run_experiments(prompts, u_values, output_csv="level1_results.csv", max_len=80):
#     model, tokenizer = load_model()
#     results = []

#     for prompt in prompts:
#         for u in u_values:
#             samples = batch_generate(model, tokenizer, prompt, u, n=20, max_len=max_len)
#             stats = token_stats(samples[0], tokenizer)
#             embeddings = embed_texts(model, tokenizer, samples)
#             sb = self_bleu(samples)
#             sim = avg_cosine_similarity(embeddings)

#             results.append({
#                 "prompt": prompt,
#                 "u": u,
#                 "self_bleu": sb,
#                 "embedding_sim": sim,
#                 **stats
#             })

#     df = pd.DataFrame(results)
#     df.to_csv(output_csv, index=False)
#     return df

def run_experiments(prompts, u_values, output_csv="level1_results.csv", max_len=80):
    model, tokenizer = load_model()
    results = []

    for prompt in prompts:
        for u in u_values:
            print(f"\n>> Running prompt='{prompt[:40]}...', u={u}")
            samples = batch_generate(model, tokenizer, prompt, u, n=20, max_len=max_len)

            stats = token_stats(samples[0], tokenizer)
            embeddings = embed_texts(model, tokenizer, samples)
            sb = self_bleu(samples)
            sim = avg_cosine_similarity(embeddings)

            results.append({
                "prompt": prompt,
                "u": u,
                "self_bleu": sb,
                "embedding_sim": sim,
                **stats
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # === automatically generate plots ===
    metrics = ["length", "unique_ratio", "entropy", "repetition", "self_bleu", "embedding_sim"]
    for metric in metrics:
        plot_metric_vs_u(df, metric)

    return df


# ---------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------
if __name__ == "__main__":
    prompts = [
        "The report concluded that",
        "In a surprising discovery, scientists found",
        "The political implications of this event are",
        "Once upon a time in a distant galaxy",
        "The patient presented with symptoms of",
        "The economic outlook for the next decade suggests",
        "Experts in cybersecurity warn that",
        "The philosophical consequences of this idea include",
        "In a groundbreaking experiment, researchers observed",
        "The unexpected behavior of the system indicated"
    ]

    # u_values = [0.00, 0.05, 0.95, 1.00]
    u_values = [round(x, 2) for x in np.linspace(0.0, 1.0, 11)]

    df = run_experiments(prompts, u_values)
    print(df.head())
