#!/usr/bin/env bash
# Whole-flow checks of r97t_open_rows.sh against the stub server and the fake probe: flows.sh <empty work directory>
# One line per flow (FLOW <name>: <gate> ok | MISMATCH want <gate>), last line FLOWS <passed>/<run>. No GPU, no
# request to a real server: every run sets R97T_OPEN_STUB=1 and writes below the work directory only.
set -uo pipefail
W=${1:?work directory}; HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../../../.." && pwd)
case "$W" in /data/logs/*|"$REPO"/*) echo "the work directory must be outside /data/logs and the repository"; exit 2 ;; esac
mkdir -p "$W"; [ -z "$(ls -A "$W")" ] || { echo "$W is not empty"; exit 2; }
DRIVER="$HERE/../r97t_open_rows.sh"; PASS=0; RUN=0
gate(){ grep -a 'R97T OPEN ROWS DONE' "$1" | tail -1 | sed 's/.*gate=//'; }
flow(){ # name preset want [VAR=value ..]: one stub run in its own directory
  local name=$1 preset=$2 want=$3 got; shift 3
  env "$@" R97T_OPEN_STUB=1 R97T_OPEN_OUT="$W/$name/out" R97T_OPEN_RCPT="$W/$name/rcpt" R97T_OPEN_CELLS=3 bash "$DRIVER" "$preset" > "$W/$name.log" 2>&1
  got=$(gate "$W/$name.log"); check "$name" "$got" "$want"; }
check(){ RUN=$((RUN + 1)); if [ "$2" = "$3" ]; then PASS=$((PASS + 1)); echo "FLOW $1: $2 ok"; else echo "FLOW $1: ${2:-none} MISMATCH want $3"; fi; }
cd "$REPO" || exit 2
flow equal flint flint:resolved-ok R97T_FAKE_SCENARIO=equal
flow equal-qwen38 qwen38 qwen38:resolved-ok R97T_FAKE_SCENARIO=equal
flow ttft-loss flint flint:kill:ttft:24 R97T_FAKE_SCENARIO=ttft-loss
flow ttft-one-boot flint flint:open:ttft:24 R97T_FAKE_SCENARIO=ttft-one-boot
flow prefill-loss flint flint:kill:prefill:7.3K,prefill:58K,prefill:197K R97T_FAKE_SCENARIO=prefill-loss
flow accept-drop flint flint:kill:accept:deep R97T_FAKE_SCENARIO=accept-drop
flow deep-drop flint flint:kill:accept:deep R97T_FAKE_SCENARIO=deep-drop
flow die-once flint flint:resolved-ok R97T_FAKE_SCENARIO=die-once
check die-once-rerun "$(cat "$W/die-once/rcpt/arm-rerun.txt" 2>/dev/null | tr '\n' ' ')" "spec-on1 "
flow die-always flint flint:server-died-spec-on1 R97T_FAKE_SCENARIO=die-always
flow fault-boot flint flint:boot-fail-warm-off R97T_STUB_FAULT=boot
flow fault-no-scale flint flint:identity-warm-on R97T_STUB_FAULT=no-scale
flow fault-leak flint flint:identity-warm-off R97T_STUB_FAULT=leak
# a second run on a finished directory boots nothing and reads the same verdict; one removed mark runs that boot only
env R97T_OPEN_STUB=1 R97T_OPEN_OUT="$W/equal/out" R97T_OPEN_RCPT="$W/equal/rcpt" R97T_OPEN_CELLS=3 bash "$DRIVER" flint > "$W/resume.log" 2>&1
check resume "$(gate "$W/resume.log") boots=$(grep -a -c '== boot' "$W/resume.log")" "flint:resolved-ok boots=0"
rm -f "$W/equal/out/DONE.spec-on4"
env R97T_OPEN_STUB=1 R97T_OPEN_OUT="$W/equal/out" R97T_OPEN_RCPT="$W/equal/rcpt" R97T_OPEN_CELLS=3 bash "$DRIVER" flint > "$W/resume-one.log" 2>&1
check resume-one "$(gate "$W/resume-one.log") boots=$(grep -a '== boot' "$W/resume-one.log" | sed 's/.*== boot \([^ ]*\).*/\1/' | tr '\n' ' ')" "flint:resolved-ok boots=spec-on4 "
# stub mode refuses the real output locations
R97T_OPEN_STUB=1 bash "$DRIVER" flint > "$W/refuse.log" 2>&1; check refuse-real-paths "rc=$? $(grep -c 'outside the real locations' "$W/refuse.log")" "rc=2 1"
echo "FLOWS $PASS/$RUN"
