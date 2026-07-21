#!/usr/bin/env bash
# Fail when one of the five shared R9700/3090 scripts diverges without an
# exact, reviewed fingerprint in KNOWN_DRIFT.tsv.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
SISTER_ROOT=${SISTER_3090:-/home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference}
MANIFEST=${KNOWN_DRIFT_FILE:-$SCRIPT_DIR/KNOWN_DRIFT.tsv}

FILES=(
  scripts/quantize/calibration_datasets.py
  scripts/quantize/run_reap.py
  scripts/eval/check_awq_scales.py
  scripts/eval/validate_capabilities.py
  scripts/bench/copyheavy_decode_bench.py
)

declare -A IN_SCOPE=()
declare -A EXPECT_R9700_HASH=()
declare -A EXPECT_SISTER_HASH=()
declare -A EXPECT_ADDITIONS=()
declare -A EXPECT_DELETIONS=()
declare -A EXPECT_REASON=()

for path in "${FILES[@]}"; do
  IN_SCOPE["$path"]=1
done

config_error() {
  printf 'CONFIG-ERROR\t%s\n' "$*" >&2
}

if [[ ! -f "$MANIFEST" ]]; then
  config_error "manifest is not a regular file: $MANIFEST"
  exit 2
fi
if [[ ! -d "$SISTER_ROOT" ]]; then
  config_error "sister repository is not a directory: $SISTER_ROOT"
  exit 2
fi

manifest_lines=$(awk 'END { print NR + 0 }' "$MANIFEST")
if (( manifest_lines > 0 )); then
  expected_header=$'path\tr9700_sha256\tsister_sha256\tsister_to_r9700_additions\tsister_to_r9700_deletions\treason'
  IFS= read -r header < "$MANIFEST" || true
  if [[ "$header" != "$expected_header" ]]; then
    config_error "invalid manifest header in $MANIFEST"
    exit 2
  fi

  line_no=1
  while IFS=$'\t' read -r path r9700_hash sister_hash additions deletions reason extra; do
    ((line_no += 1))
    [[ -z "$path$r9700_hash$sister_hash$additions$deletions$reason$extra" ]] && continue
    if [[ -n "${extra:-}" || -z "$path" || -z "$reason" ]]; then
      config_error "$MANIFEST:$line_no must contain exactly six nonempty TSV fields"
      exit 2
    fi
    if [[ -z "${IN_SCOPE[$path]:-}" ]]; then
      config_error "$MANIFEST:$line_no lists out-of-scope path: $path"
      exit 2
    fi
    if [[ -n "${EXPECT_R9700_HASH[$path]:-}" ]]; then
      config_error "$MANIFEST:$line_no duplicates path: $path"
      exit 2
    fi
    if [[ ! "$r9700_hash" =~ ^[0-9a-f]{64}$ || ! "$sister_hash" =~ ^[0-9a-f]{64}$ ]]; then
      config_error "$MANIFEST:$line_no has a malformed SHA-256"
      exit 2
    fi
    if [[ ! "$additions" =~ ^[0-9]+$ || ! "$deletions" =~ ^[0-9]+$ ]]; then
      config_error "$MANIFEST:$line_no has a non-numeric diff count"
      exit 2
    fi
    EXPECT_R9700_HASH["$path"]=$r9700_hash
    EXPECT_SISTER_HASH["$path"]=$sister_hash
    EXPECT_ADDITIONS["$path"]=$additions
    EXPECT_DELETIONS["$path"]=$deletions
    EXPECT_REASON["$path"]=$reason
  done < <(tail -n +2 "$MANIFEST")
fi

identical=0
known=0
untracked=0

for path in "${FILES[@]}"; do
  local_file=$REPO_ROOT/$path
  sister_file=$SISTER_ROOT/$path
  if [[ ! -f "$local_file" || ! -f "$sister_file" ]]; then
    config_error "missing regular file for $path (R9700=$local_file, sister=$sister_file)"
    exit 2
  fi

  local_hash=$(sha256sum -- "$local_file") || exit 2
  local_hash=${local_hash%% *}
  sister_hash=$(sha256sum -- "$sister_file") || exit 2
  sister_hash=${sister_hash%% *}
  local_mode=$(stat -c '%a' -- "$local_file") || exit 2
  sister_mode=$(stat -c '%a' -- "$sister_file") || exit 2

  if [[ "$local_hash" == "$sister_hash" ]]; then
    if [[ "$local_mode" != "$sister_mode" ]]; then
      printf 'UNTRACKED-DRIFT\t%s\tmode R9700=%s sister=%s\n' \
        "$path" "$local_mode" "$sister_mode"
      ((untracked += 1))
    elif [[ -n "${EXPECT_R9700_HASH[$path]:-}" ]]; then
      printf 'UNTRACKED-DRIFT\t%s\tfiles converged; remove stale manifest row\n' "$path"
      ((untracked += 1))
    else
      printf 'IDENTICAL\t%s\n' "$path"
      ((identical += 1))
    fi
    continue
  fi

  numstat=$(git diff --no-index --numstat -- "$sister_file" "$local_file")
  diff_rc=$?
  if (( diff_rc > 1 )); then
    config_error "git diff failed for $path with rc=$diff_rc"
    exit 2
  fi
  IFS=$'\t' read -r additions deletions _ <<< "$numstat"
  if [[ ! "$additions" =~ ^[0-9]+$ || ! "$deletions" =~ ^[0-9]+$ ]]; then
    config_error "non-text or malformed diff for $path: $numstat"
    exit 2
  fi

  if [[ "$local_mode" == "$sister_mode" \
        && "$local_hash" == "${EXPECT_R9700_HASH[$path]:-}" \
        && "$sister_hash" == "${EXPECT_SISTER_HASH[$path]:-}" \
        && "$additions" == "${EXPECT_ADDITIONS[$path]:-}" \
        && "$deletions" == "${EXPECT_DELETIONS[$path]:-}" ]]; then
    printf 'KNOWN-DRIFT\t%s\t+%s/-%s\t%s\n' \
      "$path" "$additions" "$deletions" "${EXPECT_REASON[$path]}"
    ((known += 1))
  else
    printf 'UNTRACKED-DRIFT\t%s\tactual r9700=%s sister=%s +%s/-%s modes=%s/%s\n' \
      "$path" "$local_hash" "$sister_hash" "$additions" "$deletions" \
      "$local_mode" "$sister_mode"
    if [[ -n "${EXPECT_R9700_HASH[$path]:-}" ]]; then
      printf '  expected\t%s\tsister=%s +%s/-%s\n' \
        "${EXPECT_R9700_HASH[$path]}" "${EXPECT_SISTER_HASH[$path]}" \
        "${EXPECT_ADDITIONS[$path]}" "${EXPECT_DELETIONS[$path]}"
    fi
    ((untracked += 1))
  fi
done

printf 'SUMMARY\tidentical=%d known_drift=%d untracked_drift=%d\n' \
  "$identical" "$known" "$untracked"
(( untracked == 0 ))

