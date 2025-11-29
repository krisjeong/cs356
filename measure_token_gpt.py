import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import Counter
from datasets import Dataset, load_dataset
import random
import json
RANDOM_SEED = 42
TOP_K = 50

def topk_search_unit_ranked(text, tokenizer, model, key_words):
    """
    Returns a dictionary:
        { key_word: list of (probability, rank) for each occurrence }
    """

    # Convert key words to GPT-2 tokens (BPE)
    key_tokens = []
    for w in key_words:
        bpe_token = tokenizer.tokenize(" " + w)[0]  # prepend space for word boundary
        key_tokens.append(bpe_token)
    
    # Token IDs for quick lookup
    key_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in key_tokens]

    curr = 0
    spaceless = text.split()
    results = {w: [] for w in key_words}

    while curr < len(spaceless):
        cut = random.randint(1, 100)
        end = min(curr + cut, len(spaceless))
        prompt = " ".join(spaceless[curr:end])
        curr = end

        tokens = tokenizer(prompt, return_tensors="pt").input_ids

        with torch.no_grad():
            logits = model(tokens).logits[:, -1, :]      # [1, vocab]
            probs = torch.softmax(logits[0], dim=0)      # [vocab]

            # sort descending
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            # build token_id -> rank mapping
            ranks = torch.empty_like(sorted_indices)
            ranks.scatter_(0, sorted_indices, torch.arange(sorted_indices.size(0)))

            for i, key_word in enumerate(key_words):
                tid = key_token_ids[i]
                prob = probs[tid].item()
                rank = ranks[tid].item()
                results[key_word].append((prob, rank))

    return results


def topk_search(articles, tokenizer, model, K, key_tokens):
    db = {}
    for idx, article in enumerate(articles):

        if idx % 1000 == 0:
            with open("results.json", "w", encoding="utf-8") as f:
                json.dump(db, f, indent=2)
            print("Current idx in topk search:", idx)

        title, text = article["title"], article["text"]
        out = topk_search_unit_ranked(text, tokenizer, model, key_tokens)
        db[title] = {"title" : title, "results" : out}

    return db


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    random.seed(RANDOM_SEED)

    stream = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        streaming=True)["train"]
    
    articles = []

    print("Loading stream...")
    for tide in stream:
        take = random.randint(1, 10000)
        if take == 1:
            articles.append( { "title" : tide["title"], "text" : tide["text"]})
    print(f"Stream ingested with {len(articles)} articles.")
    articles = articles[:15]
    key_tokens = ["no", "not", "nothing", "never"]
    db = topk_search(articles, tokenizer, model, TOP_K, key_tokens)
    with open("final_results.json", "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    


if __name__ == "__main__":
    main()