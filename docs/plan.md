# Study plan

Written before the main runs. The changes we made afterwards are listed at the end.

## What we want to know

1. **Does it help at a small budget?** With matched continued pretraining (same data, same order,
   same seed and schedule; 3.0B tokens at 32k context), does In-Place TTT improve RULER over the
   control, and at which context lengths and task families?
2. **Is it the test-time update?** We evaluate the trained TTT model a second time with the update
   switched off (`ttt_lr = 0`). The difference tells us how much of its score comes from updating
   the fast weights at inference, and how much from having trained with the module. The paper does
   not report this.
3. **Does the chunk size matter?** The paper's ablation on a 1.7B model prefers chunks of 512 to
   1024 tokens; the released config uses 4096. We train 512 and 1024 for 1B tokens and compare.
4. **What happens with length?** Trained at 32k, does the TTT advantage change from 4k to 64k, and
   beyond the training length?

On the side: inference throughput and memory of TTT versus the control, and a sweep of `ttt_lr`
(0, 1, 3, 6) at inference on the trained model.

## Model and recipe

Qwen3-4B-Base (36 layers). In-Place TTT as released: fast weights in the MLP down-projection of
every sixth layer (`ttt_layers = [0, 6, 12, 18, 24, 30]`), learned target projection, depthwise
Conv1D target, `ttt_lr = 3`, `ttt_chunk = 4096`. The released config lists a layer 36 that does not
exist in a 36-layer model; we drop it and the code now rejects it.

Optimizer settings follow the released `qwen3_longct.yaml`: AdamW, lr 5e-6, cosine decay to 1e-7,
2% warmup (28 steps), weight decay 0.1, gradient clipping 1.0, global batch 64 sequences of 32,768
tokens (2.1M tokens per step), 1431 steps, bf16 compute with fp32 master weights (FSDP2).

## Arms

| Arm | Training | Evaluated as | Seeds |
|---|---|---|---|
| `base` | none | Qwen3-4B-Base as released | - |
| `ttt_off` | 3.0B tokens, `ttt_mode: false` | as trained | 1, 2 |
| `ttt_on` | 3.0B tokens, `ttt_mode: true` | as trained | 1, 2 |
| `ttt_on/frozen` | (the `ttt_on` checkpoints) | `ttt_lr = 0`, fast weights never update | 1, 2 |
| `ttt_on/lr{1,6}` | (the `ttt_on` seed-1 checkpoint) | `ttt_lr` 1 and 6 at inference | 1 |
| `chunk512`, `chunk1024` | 1.0B tokens (477 steps), `ttt_chunk` 512 / 1024 | as trained | 1 |

The 1B-token checkpoints of `ttt_on` and `ttt_off` (saved at step 477) are the chunk-4096 and
control references for the chunk comparison. `ttt_on` and `ttt_off` with the same seed see the
same batches; the training script checks this at step 1 by hashing the first micro-batch on every
rank.

We skip a "base + TTT at inference only" arm: with the zero-initialized Conv1D the fast-weight
update is identically zero, so it would equal `base`.

## Data

3.25B tokens packed into 32,736-token sequences (`inplace_ttt_study/data/build_corpus.py`):
PG-19 books 1.3B, arXiv 0.75B, permissively licensed GitHub code grouped by repository 0.9B, and
synthetic retrieval documents 0.3B. Documents longer than a sequence continue into the next. The
manifest records dataset revisions and counts. 64 sequences are held out for loss checks.

## Evaluation

OpenCompass RULER, the framework the paper used, through `inplace_ttt_study/eval`: 13 tasks (8
needle-in-a-haystack variants, variable tracking, common and frequent word extraction, SQuAD and
HotpotQA), greedy decoding, 64 generated tokens, base-model prompting. Lengths 4k, 8k, 16k, 32k
and 64k, 500 samples per task for every arm (100 for the `base` calibration run). RULER instances
are generated with a fixed seed and the shared Qwen3 tokenizer, so every arm answers the same
instances and comparisons are paired.

Qwen3-4B's native window is 32k. The main 64k setting is YaRN with factor 2 (`rope_scaling` as
the Qwen3 model card recommends), applied the same way to every arm; plain RoPE at 64k is reported
as a secondary setting. Evaluation has to run at batch size 1, because the released inference code
applies the fast-weight update only to the first sequence of a batch.

## How we will read the results

* Per arm, seed, length and task: accuracy with a 95% bootstrap confidence interval over
  instances. Per length: the mean over the 13 tasks (the paper's convention), with a paired
  bootstrap over instances for differences between arms.
* We will say **TTT helps at length L** if the paired difference `ttt_on - ttt_off`, averaged over
  seeds, has a 95% interval excluding zero and the same sign for both seeds. The difference
  between the two seeds of each arm is our noise floor.
* **Attribution** at each length: `update = ttt_on - ttt_on/frozen` (effect of the test-time
  update) and `training = ttt_on/frozen - ttt_off` (effect of training with the module), each with
  intervals. A large `update` and small `training` would support the paper's mechanism; the
  reverse would mean the gain comes from the extra parameters and training dynamics rather than
  from adapting at test time.
* Chunk size: the three 1B-token TTT checkpoints and the 1B control at 32k and 64k, same rule.
* Training curves (loss, gradient norm, tokens) come from `metrics.jsonl`; held-out loss of every
  checkpoint from `scripts/check_ckpt.sh`.
* Inference cost: time and peak memory of `ttt_on` and `ttt_off` at each length.

## If things go wrong

* If the TTT arm diverges (loss above the control by more than 0.3 after warmup, or NaN), the
  released recipe is unstable at this budget; that is a finding in itself, and a stabilized variant
  (lower `ttt_lr`) is added as a clearly labelled extra arm rather than replacing the main one.
* If a 3B arm turns out to be too slow, seed 2 of both arms is reduced to 2B tokens and the report
  says so.
* No arm, metric or length is dropped from the report because of its result.

## Differences from the paper's setup

3.0B tokens instead of about 20B (stage 1) plus 15B (stage 2); the 32k stage only; a different,
public data mixture; the inner learning rate and chunk size are the released values, which the
paper does not state for Qwen3-4B; 2% warmup from the released config (the paper lists none for
continual training).

## Changes made after writing this plan

* Before any main run: the paper's Appendix 9.2 rescales any per-chunk fast-weight delta whose
  Frobenius norm exceeds tau (1e-5 for Qwen3-4B) at inference; the released inference code has no
  clipping. Patch 0003 adds it as an opt-in `ttt_clip` config field. The main evaluations of
  `ttt_on` stay as released (no clipping); `ttt_on/clip` with tau = 1e-5 is added as an extra
  evaluated variant at 32k and 64k, and per-chunk delta norms are logged on held-out sequences so
  the choice of tau can be assessed.
* After the seed-1 evaluation: the synthetic retrieval documents in the corpus taught both trained
  arms to answer some RULER prompts by starting a new "Question:" block (see results.md). The
  failure occurs almost only on `niah_single_2` and `niah_multikey_1`, and the two arms retain the
  habit to different degrees at different lengths, which dominates their difference on those
  tasks. The 13-task analysis stays primary; a sensitivity analysis that excludes those two tasks
  (`report.py --exclude-tasks`) is reported alongside it for every comparison. No other change.
