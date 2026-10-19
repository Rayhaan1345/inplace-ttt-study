"""Synthetic retrieval documents for long-context continued pretraining.

Two formats, mixed 50/50, both ending in question/answer lines so that the LM objective
rewards retrieving the planted information:

* ``kv``     - a long list of ``key: value`` records followed by queries for a few keys.
* ``needle`` - a natural-text passage (taken from the books stream) with short fact
               sentences planted at random positions, followed by questions about them.
"""
import random
import string

CITIES = ["Oslo", "Lima", "Hanoi", "Accra", "Perth", "Quito", "Cairo", "Dakar", "Bergen", "Tunis",
          "Kyoto", "Riga", "Davao", "Malmo", "Leeds", "Pune", "Arica", "Basel", "Cusco", "Ghent"]
OBJECTS = ["vault", "archive", "greenhouse", "observatory", "lighthouse", "workshop", "library",
           "hangar", "cellar", "studio", "depot", "annex", "pavilion", "reactor", "silo"]


def _key(rng):
    return "".join(rng.choices(string.ascii_lowercase + string.digits, k=8))


def kv_document(rng, n_records):
    """Key/value table with 4 queries at the end. ~n_records * 7 tokens."""
    keys = [_key(rng) for _ in range(n_records)]
    values = [rng.randint(100000, 999999) for _ in range(n_records)]
    lines = [f"{k}: {v}" for k, v in zip(keys, values)]
    asked = rng.sample(range(n_records), 4)
    qa = [f"Question: What is the value of key {keys[i]}?\nAnswer: {values[i]}" for i in asked]
    return "Records:\n" + "\n".join(lines) + "\n\n" + "\n\n".join(qa)


def needle_document(rng, passage, n_facts):
    """Plant n_facts sentences into `passage` (split on sentence ends), then ask about each."""
    sentences = passage.replace("\n", " ").split(". ")
    facts, questions = [], []
    for _ in range(n_facts):
        city, obj, code = rng.choice(CITIES), rng.choice(OBJECTS), rng.randint(1000, 9999)
        facts.append(f"The access code for the {obj} in {city} is {code}.")
        questions.append(f"Question: What is the access code for the {obj} in {city}?\nAnswer: {code}")
    for fact in facts:
        sentences.insert(rng.randint(0, len(sentences)), fact)
    return ". ".join(sentences) + "\n\n" + "\n\n".join(questions)


def iter_synthetic(passages, seed, min_tokens=8_000, max_tokens=30_000, chars_per_token=4):
    """Yield synthetic documents forever; `passages` is an iterator of long natural texts."""
    rng = random.Random(seed)
    while True:
        target = rng.randint(min_tokens, max_tokens)
        if rng.random() < 0.5:
            yield kv_document(rng, n_records=target // 7)
        else:
            passage = next(passages)[: target * chars_per_token]
            yield needle_document(rng, passage, n_facts=rng.randint(3, 8))
