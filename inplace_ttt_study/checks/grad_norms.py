"""Gradient norms by parameter group for one training batch (one GPU, no FSDP).

Loads a checkpoint with the training-time classes, runs a forward/backward pass on held-out
sequences and reports the gradient norm of the TTT parameters (`ttt_conv`, `ttt_proj`), of the
down-projections of the TTT layers, and of everything else, before any clipping. This shows which
parameters produce the large gradient norms seen in the TTT arm's training log.

    python -m inplace_ttt_study.checks.grad_norms --ckpt <hf_ckpt> --heldout runs/data/corpus/heldout.jsonl \
        --n 2 --seq-len 16384 --out <hf_ckpt>/grad_norms.json
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ["MODELING_BACKEND"] = "hf"
    import hf_models.hf_qwen3.modeling_qwen3 as train_mod

    model = train_mod.Qwen3ForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16,
                                                       attn_implementation="flash_attention_2").cuda()
    model.gradient_checkpointing_enable()
    model.train()
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    with open(args.heldout) as f:
        texts = [json.loads(next(f))["content_split"] for _ in range(args.n)]
    ttt_layers = set(getattr(model.config, "ttt_layers", [])) if getattr(model.config, "ttt_mode", False) else set()
    groups = {"ttt_conv": [], "ttt_proj": [], "down_proj_ttt_layers": [], "other": []}
    for name, p in model.named_parameters():
        if ".ttt_conv" in name:
            groups["ttt_conv"].append(p)
        elif ".ttt_proj" in name:
            groups["ttt_proj"].append(p)
        elif ".down_proj" in name and any(f".layers.{i}." in name for i in ttt_layers):
            groups["down_proj_ttt_layers"].append(p)
        else:
            groups["other"].append(p)
    losses = []
    for text in texts:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, : args.seq_len].cuda()
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss / args.n
        loss.backward()
        losses.append(loss.item() * args.n)
    norms = {g: float(torch.sqrt(sum((p.grad.float().norm() ** 2 for p in ps if p.grad is not None), torch.zeros((), device="cuda")))) for g, ps in groups.items()}
    norms["total"] = float(sum(v ** 2 for v in norms.values()) ** 0.5)
    report = dict(ckpt=args.ckpt, n=args.n, seq_len=args.seq_len, mean_loss=sum(losses) / len(losses),
                  params={g: sum(p.numel() for p in ps) for g, ps in groups.items()}, grad_norms=norms)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
