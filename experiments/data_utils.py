# experiments/data_utils.py

from datasets import load_dataset
import random

def sample_wikipedia_prompts(
    num_prompts=10,
    min_chars=50,
    max_chars=200,
    seed=42
):
    """
    Sample random snippets from Wikipedia as prompts.
    Each prompt is a substring from an article's text.
    """
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

        # random slice inside [min_chars, max_chars]
        start = random.randint(0, max(0, len(text) - max_chars))
        end = start + random.randint(min_chars, max_chars)
        snippet = text[start:end].strip()

        # avoid super weird junk
        if len(snippet.split()) < 5:
            continue

        prompts.append(snippet)

    return prompts
