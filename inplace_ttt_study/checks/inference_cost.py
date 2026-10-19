"""Inference cost of a checkpoint at several context lengths on one GPU.

Times a full-sequence prefill plus 64 greedy tokens with the inference-time classes (the same
path OpenCompass uses) and records peak memory. Run it on a TTT checkpoint and on the control to
quantify the overhead of the fast-weight updates.

    python -m inplace_ttt_study.checks.inference_cost --ckpt <hf_ckpt> --heldout runs/data/corpus/heldout.jsonl \
        --lengths 4096 8192 16384 32768 65536 --out <hf_ckpt>/inference_cost.json
"""
import argparse
import json
import time

import torch
from transformers import AutoTokenizer


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import inference_model.hf_qwen3.modeling_qwen3 as infer_mod

    model = infer_mod.Qwen3ForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16,
                                                       attn_implementation="flash_attention_2").cuda().eval()
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    with open(args.heldout) as f:
        text = " ".join(json.loads(next(f))["content_split"] for _ in range(3))
    ids_all = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    rows = []
    for L in args.lengths:
        ids = ids_all[:, :L]
        times = []
        for _ in range(args.repeats):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            model.generate(input_ids=ids, max_new_tokens=64, min_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize(); times.append(time.time() - t0)
        rows.append(dict(length=L, seconds=min(times), peak_gb=torch.cuda.max_memory_allocated() / 1e9))
        print(rows[-1], flush=True)
    report = dict(ckpt=args.ckpt, ttt_mode=bool(getattr(model.config, "ttt_mode", False)),
                  ttt_chunk=getattr(model.config, "ttt_chunk", None), rows=rows)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
