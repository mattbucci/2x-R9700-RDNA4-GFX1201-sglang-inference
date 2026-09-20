#!/bin/bash
# docker_sandbox.sh -- in-container wrapper for the Docker rollout mode (run_rollouts.py --docker).
#
#   bash docker_sandbox.sh <uid> <gid> <user> <instance_id> <port> <timeout> -- <cmd> [args...]
#
# Runs as root inside the official, unmodified SWE-bench instance image
# (sweb.eval.x86_64.<instance_id>) that score_docker.py later scores the patch
# in -- so the agent sees exactly the environment the harness will test in
# (same python, same installed deps, same pre_install edits). The container has
# no network (`--network none`); the SGLang server is reached through the
# bind-mounted unix socket /run/swebench-bridge.sock via docker_bridge.py, so
# the scaffold's configured http://127.0.0.1:<port> keeps working while web
# tools, pip and GitHub fail immediately. The scaffold toolchain and its
# config/state dirs are bind-mounted from the host under the host user's $HOME
# (read-only for binaries and profiles, read-write for session stores).
#
# Root phase:
#   1. work tree: /testbed moves to /data/swebench-work/<instance_id> -- the
#      path every audit keys sessions on (audit_predictions.py WORK_ROOTS,
#      audit_git_peek.py) -- and /testbed becomes a symlink to it so the
#      env's editable install (`/testbed` in *.egg-link / *.pth) still resolves.
#   2. git: the image's .git is the full upstream pack (327K objects on django,
#      release tags and refs/heads/main past the base commit, i.e. the fix
#      itself, resolvable by sha or ref). It is replaced by a fresh repository
#      with the current tree as its single commit: no history, no refs, no
#      unreachable objects. Untracked build artefacts stay untracked (the
#      commit respects .gitignore), and pre_install edits are already part of
#      the tree, so `git diff` afterwards is exactly the agent's edits.
#   3. the tree and the testbed conda env are chowned to the host uid, a passwd
#      entry for that uid points at $HOME, and the process drops to the host
#      uid with setpriv -- everything the agent writes into the rw state
#      mounts (opencode.db, pi/omp/prime/dcode sessions) stays host-owned.
#   4. docker_bridge.py starts and /health is checked through it (exit 97
#      "bridge check failed" if the server is not reachable: an infra
#      failure, never a model verdict).
# User phase (re-exec'd via setpriv):
#   5. `timeout -s KILL <timeout> <cmd...> < /sandbox/prompt.md` -- rc 124 on
#      expiry, the same contract as the host-side SIGKILL in sandbox mode (GNU
#      timeout reports 137 when it had to send KILL itself, so that is mapped
#      to 124 when the wall clock confirms the expiry; a 137 before the
#      deadline stays 137). The task prompt is the mounted read-only file on
#      the scaffold's stdin, never an argument: argv is visible to the agent's
#      own shell, and a `pkill -f <phrase from the issue text>` (django-11422,
#      2026-09-19) killed the scaffold, `timeout` and this script in one go
#      (rc 143, no diff). Without the mount, stdin is /dev/null as before.
#   6. scratch dirs stripped, `git add -A && git diff --cached` written to
#      /out/model.diff (bytes; the harness decodes tolerantly), rc to /out/rc,
#      and the script exits with the agent's rc.
set -u
log() { echo "[docker_sandbox $(date +%H:%M:%S)] $*" >&2; }

if [ "${1:-}" = "--agent" ]; then
  # ---- user phase ----
  shift
  IID=$1; TIMEOUT=$2; shift 2
  [ "${1:-}" = "--" ] && shift
  WORK=/data/swebench-work/$IID
  cd "$WORK" || { log "prep failed: $WORK missing in user phase"; exit 96; }
  STDIN=/dev/null
  [ -r /sandbox/prompt.md ] && STDIN=/sandbox/prompt.md
  log "agent start uid=$(id -u) cwd=$PWD python=$(command -v python) node=$(command -v node || echo none) timeout=${TIMEOUT}s prompt=$([ "$STDIN" = /dev/null ] && echo argv || echo "stdin $(wc -c < "$STDIN")B")"
  t0=$(date +%s)
  timeout -s KILL "$TIMEOUT" "$@" < "$STDIN"
  rc=$?
  elapsed=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 137 ] && [ "$elapsed" -ge "$TIMEOUT" ]; then
    log "agent killed by timeout (rc 137 after ${elapsed}s >= ${TIMEOUT}s) -> rc 124"
    rc=124
  fi
  log "agent exit rc=$rc after ${elapsed}s"
  rm -rf .claw .opencode .cache .pi .omp .prime .deepagents
  git add -A >/dev/null 2>&1
  git diff --cached > /out/model.diff 2>/out/diff.err
  echo "$rc" > /out/rc
  log "diff $(wc -c < /out/model.diff) bytes"
  exit "$rc"
fi

# ---- root phase ----
UID_=$1; GID_=$2; USER_=$3; IID=$4; PORT=$5; TIMEOUT=$6; shift 6
[ "${1:-}" = "--" ] && shift
HOMEDIR=${HOME:-/home/$USER_}
WORK=/data/swebench-work/$IID
SOCK=/run/swebench-bridge.sock
t0=$(date +%s%N)

[ -d /testbed ] || { log "prep failed: /testbed missing in image"; exit 96; }
[ -S "$SOCK" ] || { log "bridge check failed: $SOCK is not a socket"; exit 97; }

# 1-2. work tree at the audited path, fresh single-commit repository
rm -rf /testbed/.git
mkdir -p "$(dirname "$WORK")"
mv /testbed "$WORK" && ln -s "$WORK" /testbed || { log "prep failed: mv /testbed"; exit 96; }
cd "$WORK" || exit 96
git init -q \
  && git add -A \
  && git -c user.name=eval -c user.email=eval@local -c commit.gpgsign=false \
       commit -q --no-verify -m "SWE-bench $IID base tree" \
  || { log "prep failed: git reinit"; exit 96; }
git config user.email eval@local; git config user.name eval

# 3. host uid owns what the agent must write; $HOME parents (docker-created
#    mount-point dirs) too -- mount points themselves are already host-owned
#    (ro ones raise EROFS, ignored).
sed -i "/^[^:]*:[^:]*:$UID_:/d" /etc/passwd
echo "$USER_:x:$UID_:$GID_::$HOMEDIR:/bin/bash" >> /etc/passwd
getent group "$GID_" >/dev/null || echo "$USER_:x:$GID_:" >> /etc/group
chown -R "$UID_:$GID_" "$WORK" /opt/miniconda3/envs/testbed 2>/dev/null
mkdir -p "$HOMEDIR"
find "$HOMEDIR" -maxdepth 2 -type d -exec chown -h "$UID_:$GID_" {} + 2>/dev/null
chown "$UID_:$GID_" /out 2>/dev/null

# 4. bridge: 127.0.0.1:<port> -> unix socket -> host socat -> server
/opt/miniconda3/bin/python3 /sandbox/docker_bridge.py "$PORT" "$SOCK" </dev/null &
for _ in $(seq 100); do (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break; sleep 0.1; done
health=$(/opt/miniconda3/bin/python3 -c "
import urllib.request, sys
try:
    print(urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=20).status)
except Exception as e:
    print('ERR', e)
")
[ "$health" = "200" ] || { log "bridge check failed: /health via bridge -> $health"; exit 97; }
log "prep done in $(( ($(date +%s%N) - t0) / 1000000 ))ms: tree=$WORK env=$(/opt/miniconda3/envs/testbed/bin/python --version 2>&1) bridge=:$PORT ok"

# 5-6. drop to the host uid for the agent + diff capture
export HOME="$HOMEDIR" USER="$USER_" LOGNAME="$USER_"
exec setpriv --reuid="$UID_" --regid="$GID_" --clear-groups \
  bash /sandbox/docker_sandbox.sh --agent "$IID" "$TIMEOUT" -- "$@"
