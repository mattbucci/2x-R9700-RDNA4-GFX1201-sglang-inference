#!/bin/bash
# Run 256K single-user context benchmarks across all long-context-capable models.
#
# Runs models sequentially: stop previous server, launch next, bench, repeat.
# Produces benchmarks/<slug>/results.json for each.
#
# Order chosen to minimize time-to-first-data for the primary 256K target:
#   qwen35 first (native 256K, primary target), then smaller models, then MoE.
#
# Skipped: gemma4 (4K SWA limit), gemma4-31b (8K SWA limit).
#
# Usage:
#   bash scripts/bench/bench_256k_sweep.sh               # full suite
#   bash scripts/bench/bench_256k_sweep.sh qwen35-moe    # single model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

exec bash "$REPO_DIR/scripts/bench/bench_all_models.sh" "$@"
