#!/bin/bash
# sandbox.sh — run one scaffold rollout with no network and no view of other tasks.
#
#   sandbox.sh <inst_dir> <venv_dir|-> <bridge.sock> <port> -- <cmd> [args...]
#
# Unprivileged bubblewrap sandbox (user + net namespaces, same uid, no
# capabilities):
#   * a fresh network namespace with only loopback, so web tools, curl/wget,
#     `pip download` and GitHub API calls fail immediately (ENETUNREACH / no DNS).
#     The SGLang server is reached through a unix-socket bridge: run_rollouts.py
#     runs `socat UNIX-LISTEN:<bridge.sock> TCP:127.0.0.1:<port>` on the host and
#     this script runs the mirror `socat TCP-LISTEN:<port> UNIX-CONNECT` inside,
#     so the scaffold's configured http://127.0.0.1:<port> keeps working.
#   * the work root and venv root are replaced by tmpfs with only this task's
#     tree and venv bound back in, so the repo mirrors (full upstream history)
#     and other instances' work trees (later base commits that already contain
#     this task's fix) are not visible.
#   * everything else on the host is bind-mounted read-write as before.
#
# The sandboxed process stays in the caller's session/process group, so the
# harness's SIGKILL-on-timeout of the process group still reaps it.
set -u
INST_DIR=$1; VENV_DIR=$2; SOCK=$3; PORT=$4; shift 4
[ "${1:-}" = "--" ] && shift

WORK_ROOT=$(dirname "$INST_DIR")
# --unshare-pid: bwrap becomes pid 1 of the sandbox, so the bridge socat and any
# daemon a scaffold leaves behind die with the scaffold instead of lingering.
args=(--unshare-user --unshare-net --unshare-pid --uid "$(id -u)" --gid "$(id -g)"
      --bind / / --dev-bind /dev /dev --proc /proc
      --tmpfs "$WORK_ROOT" --bind "$INST_DIR" "$INST_DIR"
      --die-with-parent)
if [ "$VENV_DIR" != "-" ] && [ -d "$VENV_DIR" ]; then
  args+=(--tmpfs "$(dirname "$VENV_DIR")" --bind "$VENV_DIR" "$VENV_DIR")
fi

exec bwrap "${args[@]}" -- /bin/bash -c '
  SOCK=$1; PORT=$2; shift 2
  socat "TCP4-LISTEN:$PORT,bind=127.0.0.1,fork,reuseaddr" "UNIX-CONNECT:$SOCK" </dev/null >/dev/null 2>&1 &
  for _ in $(seq 50); do (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break; sleep 0.1; done
  exec "$@"
' _ "$SOCK" "$PORT" "$@"
