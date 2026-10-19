"""Record the fast-weight deltas a trained TTT model applies while reading held-out text.

For each TTT layer and each chunk of each sequence, the Frobenius norm of the per-chunk delta
`dw`, of the accumulated change `W_i - W_0` and of the original down-projection `W_0` are
recorded by wrapping the inference-time MLP forward. This shows whether the updates stay bounded
with position and how the paper's clipping threshold (tau = 1e-5 for Qwen3-4B) relates to the
deltas the model actually produces.

    python -m inplace_ttt_study.checks.delta_norms --ckpt <hf_ckpt> --heldout runs/data/corpus/heldout.jsonl \
        --n 8 --seq-len 32768 --out <hf_ckpt>/delta_norms.json
"""
import argparse
import json

import torch
from transformers import AutoTokenizer


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import inference_model.hf_qwen3.modeling_qwen3 as infer_mod

    records = []

    def wrap(mlp, layer_idx):
        orig = mlp.forward

        def forward(x, t=None, past_w=None):
            out, w = orig(x, t, past_w)
            if t is not None:
                w0 = mlp.down_proj.weight.float()
                records.append(dict(layer=layer_idx, seq=seq_id, tokens=x.shape[1],
                                    w0_norm=w0.norm().item(), total_delta_norm=(w.float() - w0).norm().item()))
            return out, w

        mlp.forward = forward

    model = infer_mod.Qwen3ForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16,
                                                       attn_implementation="flash_attention_2").cuda().eval()
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "ttt_conv"):
            wrap(layer.mlp, i)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    chunk = model.config.ttt_chunk
    with open(args.heldout) as f:
        texts = [json.loads(next(f))["content_split"] for _ in range(args.n)]
    per_chunk = []
    for seq_id, text in enumerate(texts):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, : args.seq_len].cuda()
        # feed the sequence chunk by chunk with the cache so each call applies exactly one delta
        cache = None
        for start in range(0, ids.shape[1], chunk):
            out = model(input_ids=ids[:, start:start + chunk], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            for layer_idx, (_, _, w) in enumerate(cache.ttt_states[: len(model.model.layers)]):
                if w is not None:
                    w0 = model.model.layers[layer_idx].mlp.down_proj.weight.float()
                    per_chunk.append(dict(seq=seq_id, layer=layer_idx, position=start + chunk,
                                          accumulated_delta_norm=(w.float() - w0).norm().item(), w0_norm=w0.norm().item()))
    summary = {}
    for r in per_chunk:
        key = f"layer{r['layer']}@{r['position']}"
        summary.setdefault(key, []).append(r["accumulated_delta_norm"] / r["w0_norm"])
    report = dict(ckpt=args.ckpt, chunk=chunk, ttt_lr=model.config.ttt_lr, n=args.n, seq_len=args.seq_len,
                  relative_accumulated_delta_by_layer_and_position={k: sum(v) / len(v) for k, v in summary.items()},
                  per_chunk=per_chunk)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "per_chunk"}, indent=1))


if __name__ == "__main__":
    main()
