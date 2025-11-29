# experiments/model_utils.py

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import datetime
import os

def load_model(model_name="gpt2", device=None):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def generate_with_fixed_u(model, tokenizer, prompt, u, max_len=80, device="cpu"):
    """
    Production generator: fixed u, no per-step logging.
    Uses descending-sorted probabilities so that u acts as a quantile
    over probability mass (top of the distribution).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]  # (1, vocab)
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (vocab,)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=0)
            idx = torch.searchsorted(cdf, torch.tensor(u, device=device))
            # searchsorted returns len(cdf) when u==1.0; clamp to last index
            idx = torch.clamp(idx, max=cdf.numel() - 1)

            chosen_token_id = sorted_indices[idx].view(1, 1)  # (1,1)
            generated = torch.cat([generated, chosen_token_id], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def generate_with_fixed_u_debug(
    model,
    tokenizer,
    prompt,
    u,
    max_len=50,
    device="cpu",
    log_path=None,
    print_console=True
):
    """
    Debug version: logs each step, token id, etc.
    """
    def log(msg):
        if print_console:
            print(msg)
        if log_path:
            with open(log_path, "a") as f:
                f.write(msg + "\n")

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"=== LOG START ({ts}) ===\n")

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids.clone()

    log(f"\n=== GENERATING WITH u = {u} ===")
    log(f"Prompt: {prompt}\n")

    with torch.no_grad():
        for step in range(max_len):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]

            probs = torch.softmax(logits, dim=-1).squeeze(0)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=0)
            idx = torch.searchsorted(cdf, torch.tensor(u, device=device))
            idx = torch.clamp(idx, max=cdf.numel() - 1)

            chosen_token_id = sorted_indices[idx].view(1, 1)

            token_str = tokenizer.decode(chosen_token_id[0])
            max_prob = float(sorted_probs[0])

            new_generated = torch.cat([generated, chosen_token_id], dim=1)
            partial_text = tokenizer.decode(new_generated[0], skip_special_tokens=True)

            log(f"Step {step+1}")
            log(f"  u: {u}")
            log(f"  chosen token id: {chosen_token_id.item()} ({repr(token_str)})")
            log(f"  highest prob token: {max_prob:.5f}")
            log(f"  sorted index chosen: {idx.item()}")
            log(f"  Generated so far: {partial_text}")
            log("-" * 50)

            generated = new_generated

    final_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    log("\n=== FINAL OUTPUT ===")
    log(final_text)
    log("=" * 50)

    return final_text
