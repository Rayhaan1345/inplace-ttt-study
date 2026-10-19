"""Sanity checks for a saved HuggingFace checkpoint of a training arm (run on one GPU).

1. The checkpoint loads with the TTT-aware classes without missing or unexpected tensors.
2. Held-out loss is finite and is reported next to the base model's loss on the same sequences.
3. The training-time and inference-time TTT implementations agree (same loss on the same input).
4. With ttt_lr=0 (fast weights frozen) the inference model equals ttt_mode=false on the same input,
   which is the identity the lesion arm relies on.

Usage (from the repo root, `ttt` env, PYTHONPATH including third_party/In-Place-TTT):
    python -m inplace_ttt_study.checks.check_checkpoint --ckpt <hf_ckpt dir> --base runs/models/Qwen3-4B-Base \
        --heldout runs/data/corpus/heldout.jsonl --n 4 --seq-len 8192
"""
import argparse
import json

import torch
from transformers import AutoTokenizer


def load(cls, path, **overrides):
    model, info = cls.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
                                      output_loading_info=True, **overrides)
    unexpected = info["unexpected_keys"]
    if overrides.get("ttt_mode") is False:  # TTT disabled: the checkpoint's TTT tensors are unused on purpose
        unexpected = [k for k in unexpected if ".ttt_" not in k]
    assert not info["missing_keys"] and not unexpected, f"{cls.__module__}: missing={info['missing_keys']} unexpected={unexpected}"
    return model.cuda().eval()


def mean_loss(model, batches):
    losses = [model(input_ids=b, labels=b, use_cache=False).loss.item() for b in batches]
    return sum(losses) / len(losses)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--out", help="write the report as JSON to this path")
    args = ap.parse_args()

    import hf_models.hf_qwen3.modeling_qwen3 as train_mod  # training-time class (liger fused loss)
    import inference_model.hf_qwen3.modeling_qwen3 as infer_mod  # inference-time class (chunk loop + cache)

    tok = AutoTokenizer.from_pretrained(args.base)
    with open(args.heldout) as f:
        texts = [json.loads(next(f))["content_split"] for _ in range(args.n)]
    batches = [tok(t, return_tensors="pt", add_special_tokens=False).input_ids[:, : args.seq_len].cuda() for t in texts]
    report = {"ckpt": args.ckpt, "n": args.n, "seq_len": args.seq_len}

    model = load(infer_mod.Qwen3ForCausalLM, args.base)
    report["base_loss"] = mean_loss(model, batches)
    del model; torch.cuda.empty_cache()

    model = load(infer_mod.Qwen3ForCausalLM, args.ckpt)
    ttt = bool(getattr(model.config, "ttt_mode", False))
    report["ckpt_loss_inference_class"] = mean_loss(model, batches)
    if ttt:
        logits_ttt = model(input_ids=batches[0], use_cache=False).logits
        del model; torch.cuda.empty_cache()
        model = load(infer_mod.Qwen3ForCausalLM, args.ckpt, ttt_lr=0.0)
        logits_frozen = model(input_ids=batches[0], use_cache=False).logits
        report["max_abs_diff_ttt_vs_frozen"] = (logits_ttt - logits_frozen).abs().max().item()
        del model, logits_ttt; torch.cuda.empty_cache()
        model = load(infer_mod.Qwen3ForCausalLM, args.ckpt, ttt_mode=False)
        logits_off = model(input_ids=batches[0], use_cache=False).logits
        report["max_abs_diff_frozen_vs_off"] = (logits_frozen - logits_off).abs().max().item()
        del logits_frozen, logits_off
    del model; torch.cuda.empty_cache()

    model = load(train_mod.Qwen3ForCausalLM, args.ckpt)
    report["ckpt_loss_training_class"] = mean_loss(model, batches)
    report["ttt_param_norms"] = {n: round(p.float().norm().item(), 4) for n, p in model.named_parameters() if ".ttt_" in n}
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)

    assert report["ckpt_loss_inference_class"] < 10, "checkpoint loss is not sane"
    assert abs(report["ckpt_loss_inference_class"] - report["ckpt_loss_training_class"]) < 0.05, "train/infer TTT mismatch"
    if ttt:
        assert report["max_abs_diff_frozen_vs_off"] < 1e-2, "ttt_lr=0 is not equivalent to ttt_mode=false"
    print("CHECKS PASSED")


if __name__ == "__main__":
    main()
