# experiments/metrics.py

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd

# --------- Token-level stats per sequence ---------

def token_stats(text, tokenizer):
    tokens = tokenizer.encode(text)
    L = len(tokens)
    unique_ratio = len(set(tokens)) / L if L > 0 else 0

    counts = Counter(tokens)
    freqs = np.array(list(counts.values()), dtype=float)
    if L > 0:
        freqs /= L
        entropy = -(freqs * np.log(freqs)).sum()
    else:
        entropy = 0.0

    repetitions = (
        sum(tokens[i] == tokens[i-1] for i in range(1, L)) / L
        if L > 1 else 0.0
    )

    return {
        "length": L,
        "unique_ratio": unique_ratio,
        "entropy": float(entropy),
        "repetition": float(repetitions),
    }


# --------- Sequence-level diversity metrics ---------

def self_bleu(samples):
    """
    Mean BLEU of each sample against all others as references.
    Measures diversity: lower = more diverse.
    """
    if len(samples) <= 1:
        return 0.0
    scores = []
    tokenized = [s.split() for s in samples]
    for i, hyp in enumerate(tokenized):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        if not refs:
            continue
        scores.append(sentence_bleu(refs, hyp))
    return float(np.mean(scores)) if scores else 0.0


def embed_texts(model, tokenizer, samples, device="cpu"):
    """
    Use GPT-2 transformer as an encoder by averaging hidden states.
    Accepts either a full LM (with .transformer) or the bare transformer model.
    """
    embeddings = []
    with torch.no_grad():
        for s in samples:
            ids = tokenizer.encode(s, return_tensors='pt').to(device)
            # Support both GPT2LMHeadModel (has .transformer attribute) and GPT2Model
            encoder = model.transformer if hasattr(model, "transformer") else model
            hidden = encoder(ids).last_hidden_state.mean(dim=1)  # (1, hidden)
            embeddings.append(hidden.squeeze(0))
    return torch.stack(embeddings)  # (N, hidden)


def avg_pairwise_cosine(embeddings):
    """
    Average cosine similarity over all pairs of embeddings.
    """
    n = embeddings.size(0)
    if n < 2:
        return 1.0
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            sims.append(F.cosine_similarity(embeddings[i], embeddings[j], dim=0).item())
    return float(np.mean(sims))


# --------- Per-position entropy across runs (for heatmaps) ---------

def position_token_entropy(samples, tokenizer, max_len=80):
    """
    For a set of generated texts, compute for each position t in [0, max_len):
    - the entropy of the token distribution at that position across runs.

    Returns: np.array of shape (max_len,), entropies per position.
    """
    # Encode all samples as token ids
    encoded = [tokenizer.encode(s) for s in samples]

    entropies = []
    for t in range(max_len):
        # collect tokens at position t (if present)
        toks_at_t = []
        for seq in encoded:
            if t < len(seq):
                toks_at_t.append(seq[t])

        if len(toks_at_t) == 0:
            entropies.append(0.0)
            continue

        counts = Counter(toks_at_t)
        total = sum(counts.values())
        freqs = np.array(list(counts.values()), dtype=float) / total
        ent = -(freqs * np.log(freqs)).sum()
        entropies.append(float(ent))

    return np.array(entropies, dtype=float)


# --------- Helper to aggregate metrics over runs ---------

def aggregate_metrics(samples, tokenizer, embed_model, device="cpu", max_len=80):
    """
    Compute aggregate metrics over multiple samples for a given prompt/u.
    - token_stats of first sample (for a concrete sequence-level example)
    - mean self-BLEU
    - embedding similarity (avg pairwise cosine)
    - per-position entropy (for heatmaps)

    Returns:
        metrics_dict (flat)
        pos_entropy (np.array of shape (max_len,))
    """
    # basic token stats on first sample
    base_stats = token_stats(samples[0], tokenizer) if samples else {
        "length": 0,
        "unique_ratio": 0.0,
        "entropy": 0.0,
        "repetition": 0.0,
    }

    sb = self_bleu(samples)
    embeddings = embed_texts(embed_model, tokenizer, samples, device=device)
    sim = avg_pairwise_cosine(embeddings)
    pos_ent = position_token_entropy(samples, tokenizer, max_len=max_len)

    metrics = {
        **base_stats,
        "self_bleu": sb,
        "embedding_sim": sim,
    }

    return metrics, pos_ent
