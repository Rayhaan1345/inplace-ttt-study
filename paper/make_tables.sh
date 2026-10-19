#!/bin/bash
# Regenerates the appendix tables from the per-instance CSV (see README, Analysis).
# Usage: paper/make_tables.sh $STUDY_ROOT/results/ruler_instances.csv
set -euo pipefail
CSV=${1:?per-instance csv}; OUT=$(dirname "$0")/tables; mkdir -p "$OUT"
T="python -m inplace_ttt_study.analysis.latex_tables $CSV"
MAIN="base:Base ttt_off_s1:Control~s1 ttt_on_s1:TTT~s1 ttt_on_s1_frozen:TTT~s1~frozen ttt_off_s2:Control~s2 ttt_on_s2:TTT~s2 ttt_on_s2_frozen:TTT~s2~frozen"
for L in 4k 8k 16k 32k 64k; do $T --length $L --models $MAIN --out "$OUT/pertask_$L.tex"; done
$T --length 64k --models base_yarn2:Base ttt_off_s1_yarn2:Control~s1 ttt_on_s1_yarn2:TTT~s1 ttt_on_s1_frozen_yarn2:TTT~s1~frozen ttt_off_s2_yarn2:Control~s2 ttt_on_s2_yarn2:TTT~s2 ttt_on_s2_frozen_yarn2:TTT~s2~frozen --out "$OUT/pertask_64k_yarn.tex"
for L in 32k 64k; do
  $T --length $L --models "ttt_on_s1_frozen:lr~0" "ttt_on_s1_lr1:lr~1" "ttt_on_s1:lr~3" "ttt_on_s1_lr6:lr~6" "ttt_on_s1_clip:lr~3,~clip" --out "$OUT/sweep_$L.tex"
  $T --length $L --models "control_1b:Control~1B" "chunk4096:Chunk~4096" "chunk4096_frozen:4096~frozen" "chunk1024:Chunk~1024" "chunk1024_frozen:1024~frozen" "chunk512:Chunk~512" "chunk512_frozen:512~frozen" --out "$OUT/chunk_$L.tex"
  $T --length $L --models "control_1b:Control~1B" "chunk4096:TTT~1B" "chunk4096_frozen:TTT~1B~frozen" "ttt_off_s1_2b:Control~2B" "ttt_on_s1_2b:TTT~2B" "ttt_on_s1_2b_frozen:TTT~2B~frozen" "ttt_off_s1:Control~3B" "ttt_on_s1:TTT~3B" "ttt_on_s1_frozen:TTT~3B~frozen" --out "$OUT/curve_$L.tex"
done
