#!/usr/bin/env bash
# Submit all issues to GitHub via gh CLI
# Usage: bash .github/issues/submit_all.sh
# Requires: gh auth login (run once before this script)

set -euo pipefail
REPO="sea-sky-web/BEV_Track-Predict"
DIR="$(cd "$(dirname "$0")" && pwd)"

submit() {
  local file="$1" title="$2" labels="$3"
  echo "→ Submitting: $title"
  gh issue create \
    --repo "$REPO" \
    --title "$title" \
    --body-file "$file" \
    --label "$labels" 2>/dev/null \
    || gh issue create --repo "$REPO" --title "$title" --body-file "$file"
  sleep 2   # avoid secondary rate limit
}

submit "$DIR/issue_01_pretrained_backbone.md" \
  "[BUG][CRITICAL] Pretrained backbone disabled by default — primary cause of F1≈0.10" \
  "bug,critical"

submit "$DIR/issue_02_optimizer_config.md" \
  "[BUG][HIGH] Optimizer misconfiguration: SGD momentum=0.5 + OneCycleLR max_lr=0.1 causes unstable training" \
  "bug"

submit "$DIR/issue_03_data_limits.md" \
  "[BUG][HIGH] Training hard-capped at 300 frames and 3/7 views — insufficient for convergence" \
  "bug"

submit "$DIR/issue_04_projection_validation.md" \
  "[BUG][MEDIUM] Projection matrix correctness unverified — add visualization unit test" \
  "bug"

submit "$DIR/issue_05_backbone_resnet18.md" \
  "[IMPROVEMENT][HIGH] Replace ResNet-50 with ResNet-18 to reduce overfitting and training cost" \
  "enhancement"

submit "$DIR/issue_06_moda_modp_metrics.md" \
  "[IMPROVEMENT][MEDIUM] Evaluation metrics don't align with MVDet standard — add MODA/MODP" \
  "enhancement"

submit "$DIR/issue_07_augmentation.md" \
  "[IMPROVEMENT][MEDIUM] Add view-coherent data augmentation — currently zero augmentation applied" \
  "enhancement"

submit "$DIR/issue_08_confidence_fusion.md" \
  "[IMPROVEMENT][MEDIUM] Confidence fusion architecture too shallow to learn meaningful view weights" \
  "enhancement"

submit "$DIR/issue_09_ci_forward_test.md" \
  "[IMPROVEMENT][LOW] CI smoke test only checks imports — extend to true end-to-end forward pass" \
  "enhancement"

echo ""
echo "✓ All 9 issues submitted to https://github.com/$REPO/issues"
