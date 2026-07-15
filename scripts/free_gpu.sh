#!/bin/bash
# Free the GPUs between serving runs and clear leaked RCCL IPC state.
#
# Hard-killing a TP=2 server does not call destroy_process_group(), so the RCCL
# communicator's IPC resources leak: orphaned psm_* segments pile up in /dev/shm.
# A relaunch within a few seconds can then fault the new RCCL init on a stale IPC
# page -> amdgpu [gfxhub] page fault -> "Fatal Python error: Aborted" -> a rank's
# scheduler dies (exit -6, SIGABRT; NOT the OOM killer's -9). It is intermittent
# (~20% of back-to-back boots) and the GPU recovers, but this settles it.
#
# Safe by construction: kills only by PID (never a bare 'sglang' -f pattern, which
# would match this repo's own path and self-kill), and prunes only shm segments
# that NO live process holds open -> cannot disturb a concurrent calibration/serve.
set -u

echo "[free_gpu] killing sglang servers + TP workers by PID"
for p in $(pgrep -f 'sglang.launch_server' 2>/dev/null); do kill "$p" 2>/dev/null; done
sleep 4
for p in $(pgrep -f 'AI/models/' 2>/dev/null); do kill "$p" 2>/dev/null; done

echo "[free_gpu] waiting for VRAM to drain"
for i in $(seq 1 30); do
  u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0/&&/Used/{gsub(/[^0-9]/,"",$NF);print $NF;exit}')
  [ -n "$u" ] && [ "$u" -lt 1073741824 ] 2>/dev/null && break
  sleep 3
done

# Prune leaked IPC segments — ONLY those no process currently holds open.
pruned=0
if command -v fuser >/dev/null 2>&1; then
  for seg in /dev/shm/psm_* /dev/shm/nccl-* /dev/shm/rccl-*; do
    [ -e "$seg" ] || continue
    if ! fuser "$seg" >/dev/null 2>&1; then rm -f "$seg" && pruned=$((pruned+1)); fi
  done
  echo "[free_gpu] pruned $pruned orphaned /dev/shm IPC segment(s)"
else
  echo "[free_gpu] fuser not available — skipping shm prune (install psmisc to enable)"
fi

# Settle so KFD/ROCr fully reclaims GPU + IPC contexts before the next boot.
sleep 8
echo "[free_gpu] done. GPU0 used: $(rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0/&&/Used/{print $NF; exit}') B"
