"""Build the packed long-context corpus used by every training arm.

Public Hugging Face datasets are streamed, tokenized with the study tokenizer, and concatenated
(EOS-separated) into sequences of exactly ``seq_len - margin`` tokens. Each sequence is decoded
back to text and written as one jsonl line ``{"content_split": ..., "source": ...}``; VeOmni
re-tokenizes the text at load time, and the margin keeps that length within ``seq_len``.
Output layout: ``<out>/train/shard-*.jsonl`` (point VeOmni at ``<out>/train``), ``<out>/heldout.jsonl``
(sequences never used for training, for loss checks) and ``<out>/manifest.json``.
Documents longer than one sequence (books, repositories) continue into the next sequence, so the
model sees long contiguous contexts. Finally all sequences are shuffled with a fixed seed and
split into ``--shards`` files, and a manifest with per-source token counts is written.

Example (about 3.25B tokens, ~2 h on 12 CPUs):
    python -m inplace_ttt_study.data.build_corpus --out runs/data/corpus \
        --tokenizer runs/models/Qwen3-4B-Base --budget books=1.3e9 arxiv=0.75e9 code=0.9e9 synthetic=0.3e9
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import time

from .synthetic import iter_synthetic

SOURCES = {
    "books": dict(path="emozilla/pg19", split="train", text="text"),
    "arxiv": dict(path="ccdv/arxiv-summarization", name="document", split="train", text="article"),
    # ships a loading script, so its parquet files are read directly instead of running the script
    "code": dict(path="codeparrot/github-code-clean", files="data/train-*.parquet", text="code"),
}
PERMISSIVE_LICENSES = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "unlicense", "cc0-1.0"}


def stream(source, seed, buffer_size=1000):
    """Stream raw documents of one source. Code keeps file order so repositories stay contiguous."""
    from datasets import load_dataset

    spec = SOURCES[source]
    if "files" in spec:
        ds = load_dataset("parquet", data_files=f"hf://datasets/{spec['path']}/{spec['files']}", split="train",
                          streaming=True)
    else:
        ds = load_dataset(spec["path"], spec.get("name"), split=spec["split"], streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=1 if source == "code" else buffer_size)
    if source != "code":
        for row in ds:
            yield row[spec["text"]]
        return
    repo, files = None, []
    for row in ds:
        if row["license"] not in PERMISSIVE_LICENSES:
            continue
        if row["repo_name"] != repo and files:
            yield "\n\n".join(files)
            files = []
        repo = row["repo_name"]
        files.append(f"# {row['path']}\n{row['code']}")
    if files:
        yield "\n\n".join(files)


def documents(source, seed, tokenizer):
    if source != "synthetic":
        return stream(source, seed)
    passages = (p for p in stream("books", seed + 1) if len(p) > 50_000)
    return iter_synthetic(passages, seed)


def pack(docs, tokenizer, length):
    """Concatenate EOS-terminated token streams into exact `length`-token sequences."""
    buf = []
    for text in docs:
        buf.extend(tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id])
        while len(buf) >= length:
            yield buf[:length]
            buf = buf[length:]


def build_source(source, budget, args):
    """Write one source's sequences to <out>/parts/<source>.jsonl until `budget` tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    length = args.seq_len - args.margin
    path = os.path.join(args.out, "parts", f"{source}.jsonl")
    stats = dict(source=source, sequences=0, tokens=0, max_retokenized=0, over_limit=0)
    t0 = time.time()
    with open(path, "w") as f:
        for ids in pack(documents(source, args.seed, tokenizer), tokenizer, length):
            text = tokenizer.decode(ids, skip_special_tokens=False)
            n = len(tokenizer.encode(text, add_special_tokens=False))
            stats["max_retokenized"] = max(stats["max_retokenized"], n)
            stats["over_limit"] += n > args.seq_len
            f.write(json.dumps({"content_split": text, "source": source}) + "\n")
            stats["sequences"] += 1
            stats["tokens"] += min(n, args.seq_len)
            if stats["sequences"] % 500 == 0:
                print(f"[{source}] {stats['tokens']/1e9:.3f}B tokens, {time.time()-t0:.0f}s", flush=True)
            if stats["tokens"] >= budget:
                break
    stats["seconds"] = round(time.time() - t0)
    return stats


def shuffle_into_shards(out, sources, n_shards, seed, n_heldout):
    """Globally shuffle all sequences (by file offset); write a held-out file, then round-robin shards."""
    index = []
    for source in sources:
        path = os.path.join(out, "parts", f"{source}.jsonl")
        with open(path, "rb") as f:
            offset = 0
            for line in f:
                index.append((source, offset))
                offset += len(line)
    random.Random(seed).shuffle(index)
    handles = {s: open(os.path.join(out, "parts", f"{s}.jsonl"), "rb") for s in sources}
    os.makedirs(os.path.join(out, "train"), exist_ok=True)
    shards = [open(os.path.join(out, "train", f"shard-{i:03d}.jsonl"), "wb") for i in range(n_shards)]
    heldout = open(os.path.join(out, "heldout.jsonl"), "wb")
    for i, (source, offset) in enumerate(index):
        handles[source].seek(offset)
        (heldout if i < n_heldout else shards[i % n_shards]).write(handles[source].readline())
    for h in list(handles.values()) + shards + [heldout]:
        h.close()
    for s in sources:
        os.remove(os.path.join(out, "parts", f"{s}.jsonl"))
    return len(index) - n_heldout


def dataset_revisions():
    """Commit hash of each source dataset at build time, recorded in the manifest."""
    from huggingface_hub import HfApi

    return {name: HfApi().dataset_info(spec["path"]).sha for name, spec in SOURCES.items()}


def parse_budget(items):
    budget = {}
    for item in items:
        name, value = item.split("=")
        budget[name] = int(float(value))
    return budget


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer path (the base model directory)")
    ap.add_argument("--budget", nargs="+", required=True, help="tokens per source, e.g. books=1.3e9")
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--margin", type=int, default=32, help="tokens of slack for re-tokenization")
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--heldout", type=int, default=64, help="sequences reserved in heldout.jsonl")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    budget = parse_budget(args.budget)
    assert set(budget) <= set(SOURCES) | {"synthetic"}, f"unknown source in {budget}"
    os.makedirs(os.path.join(args.out, "parts"), exist_ok=True)

    with mp.Pool(len(budget)) as pool:
        stats = pool.starmap(build_source, [(s, b, args) for s, b in budget.items()])
    total = shuffle_into_shards(args.out, list(budget), args.shards, args.seed, args.heldout)
    manifest = dict(args=vars(args), sources=SOURCES, revisions=dataset_revisions(),
                    licenses=sorted(PERMISSIVE_LICENSES), per_source=stats,
                    train_sequences=total, heldout_sequences=args.heldout, tokens=sum(s["tokens"] for s in stats))
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
