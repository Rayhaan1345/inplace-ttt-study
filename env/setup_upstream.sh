#!/usr/bin/env bash
# Clone ByteDance-Seed/In-Place-TTT at the pinned commit into third_party/ and apply patches/*.patch
# (layer-index validation, per-step metrics, rank data check). Re-running resets to a clean state.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
DEST=third_party/In-Place-TTT
if [ ! -d "$DEST/.git" ]; then
  git clone --quiet https://github.com/ByteDance-Seed/In-Place-TTT.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet --force "$COMMIT"
for p in patches/*.patch; do git -C "$DEST" apply --3way "../../$p"; done
echo "upstream at $COMMIT with $(ls patches/*.patch | wc -l | tr -d ' ') patches applied"
