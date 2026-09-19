# OCI image: build, run, and hardening

Moved verbatim from the top-level README on 2026-09-10. `Dockerfile`, `docker/build-sglang.sh`,
`docker/entrypoint.sh`, and `docker/secure-launch.py` are the sources of truth; this page is the operator guide.

`Dockerfile` builds the ROCm 7.2/v0.5.20 stack without a GPU. The base images, SGLang commit, Rust toolchain, and downloaded installer checksums are pinned. Python/Conda transitive artifacts and live apt repositories are not fully hash-locked, so this is version-constrained rather than bit-reproducible. GitHub Actions verifies PR builds with a read-only token and, on trusted main-branch pushes, promotes the exact inspected candidate digest to a full-commit `sha-*` tag at `ghcr.io/<owner>/sglang-rdna4`; pin deployments by digest because registry tags remain mutable. If a version alias is needed, create it in a trusted release workflow by promoting an already verified digest—do not rebuild from a tag.

The image defaults to `GPU_IDS=0`; `TP` defaults to the comma-separated `GPU_IDS` count. For example, use `GPU_IDS=0 TP=1` or `GPU_IDS=0,1 TP=2`. It exports the selection through `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL`, and `CUDA_VISIBLE_DEVICES`, and rejects TP values larger than the selection. The TP=1-only `SGLANG_RDNA4_DISABLE_STORE_CACHE=1` fallback avoids the RDNA4 JIT KV-store crash; TP=2 leaves the store-cache path unchanged.

Host-side AMD device selection (`/dev/kfd` plus selected `/dev/dri` render nodes, or AMD CDI) determines whether one or two GPUs are available; the image is generic. `GPU_IDS` is scheduling configuration, not an isolation boundary. P2P settings apply only to two-GPU TP workloads and still require the kernel/IOMMU prerequisites in the [top-level README](../README.md#current-stack).

Hardened preset launches fail closed unless both a client API key and a distinct admin key of at least 32 characters are supplied from read-only files. The parent directory below prevents other host users from traversing to the files; mode `0404` lets the image's unprivileged UID 10001 read each file after it is bind-mounted without placing the secret value in `docker inspect` or process arguments. Inspect output still reveals the environment names and host bind-mount paths:

```bash
install -d -m 0700 ./sglang-secrets
python - <<'PY'
import secrets
from pathlib import Path

directory = Path("sglang-secrets")
for filename in ("api-key", "admin-api-key"):
    path = directory / filename
    path.write_text(secrets.token_urlsafe(48) + "\n", encoding="ascii")
    path.chmod(0o404)
PY
api_secret="$(pwd)/sglang-secrets/api-key"
admin_secret="$(pwd)/sglang-secrets/admin-api-key"
```

Run a preset explicitly so `TP` reaches SGLang's `--tensor-parallel-size`. The numeric supplemental groups grant the image's unprivileged UID 10001 access to only the passed device nodes:

```bash
# One GPU (replace renderD128 and model path for the host).
docker run --rm \
  -p 127.0.0.1:8000:23334 \
  --device /dev/kfd:/dev/kfd:rw \
  --device /dev/dri/renderD128:/dev/dri/renderD128:rw \
  --group-add "$(stat -c '%g' /dev/kfd)" \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  --cap-drop=ALL --security-opt=no-new-privileges:true \
  --pids-limit 4096 --shm-size 16g \
  -e SGLANG_API_KEY_FILE=/run/secrets/sglang-api-key \
  -e SGLANG_ADMIN_API_KEY_FILE=/run/secrets/sglang-admin-api-key \
  --mount type=bind,src="$api_secret",dst=/run/secrets/sglang-api-key,readonly \
  --mount type=bind,src="$admin_secret",dst=/run/secrets/sglang-admin-api-key,readonly \
  -e MODELS_DIR=/models --mount type=bind,src=/path/to/models,dst=/models,readonly \
  ghcr.io/<owner>/sglang-rdna4@sha256:<image-digest> \
  scripts/launch.sh coder-30b

# Two GPUs; the selected render nodes and GPU_IDS must agree.
docker run --rm \
  -p 127.0.0.1:8000:23334 \
  --device /dev/kfd:/dev/kfd:rw \
  --device /dev/dri/renderD128:/dev/dri/renderD128:rw \
  --device /dev/dri/renderD129:/dev/dri/renderD129:rw \
  --group-add "$(stat -c '%g' /dev/kfd)" \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  --group-add "$(stat -c '%g' /dev/dri/renderD129)" \
  --cap-drop=ALL --security-opt=no-new-privileges:true \
  --pids-limit 4096 --shm-size 16g \
  -e SGLANG_API_KEY_FILE=/run/secrets/sglang-api-key \
  -e SGLANG_ADMIN_API_KEY_FILE=/run/secrets/sglang-admin-api-key \
  --mount type=bind,src="$api_secret",dst=/run/secrets/sglang-api-key,readonly \
  --mount type=bind,src="$admin_secret",dst=/run/secrets/sglang-admin-api-key,readonly \
  -e GPU_IDS=0,1 -e TP=2 -e MODELS_DIR=/models \
  --mount type=bind,src=/path/to/models,dst=/models,readonly \
  ghcr.io/<owner>/sglang-rdna4@sha256:<image-digest> \
  scripts/launch.sh coder-30b
```

Keep the private `/dev/shm` allocation bounded; do not replace it with `--ipc=host`. To validate the image with a read-only root filesystem, supply writable JIT/IPC locations, for example:

```bash
--read-only \
--tmpfs /tmp:rw,nodev,nosuid,size=8g,uid=10001,gid=10001,mode=1777 \
--tmpfs /home/sglang/.cache:rw,nodev,nosuid,size=8g,uid=10001,gid=10001,mode=0700
```

The image sets `SGLANG_TRUST_REMOTE_CODE=0`, disables unauthenticated metrics and custom serialized logit processors, and bounds the request queue at 32 by default. Its hardened preset path also rejects LoRA tensor deserialization, tool servers, KV-event/debug publishers, the scripted test runtime, remote-instance and ModelExpress transports, MoE/elastic backends, multi-node/disaggregated modes, and alternate gRPC/bootstrap listeners. Single-node PyTorch/RCCL bootstrap traffic is forced onto loopback even though the keyed HTTP API listens inside the container on `0.0.0.0`. It patches v0.5.16 to keep credentials out of logs, status responses, dumps, and WebSocket authentication gaps. Remote-URL and local-path multimodal inputs are disabled by default across the shared and model-specific loaders; inline/base64 media remains available. Enable remote model code only for a reviewed, immutable checkpoint with `-e SGLANG_TRUST_REMOTE_CODE=1`; a read-only model mount does not make its Python code safe. Tune the queue with `SGLANG_MAX_QUEUED_REQUESTS`. If metrics are needed, set `SGLANG_ENABLE_METRICS=1` only on a private monitoring network.

Do not publish SGLang directly to an untrusted network. Docker's loopback binding above is deliberate, but it does not stop a container on the same bridge network from reaching the container IP; do not share that network with untrusted workloads. For remote clients, use a private network plus a TLS reverse proxy that authenticates, rate-limits, caps request bodies, denies management routes (including `/server_info` and `/get_server_info`), and blocks `/v1/realtime` unless WebSocket authentication is explicitly supported. SGLang v0.5.16 exempts `/health*` and `/metrics*` from API-key checks. If URL or local-path media is explicitly enabled with `SGLANG_ALLOW_REMOTE_MEDIA=1` or `SGLANG_ALLOW_LOCAL_MEDIA=1`, treat that as a trust-boundary change: restrict proxy routes and container egress to prevent SSRF, local-file disclosure, and unbounded downloads. Do not embed credentials in model/tool URLs or config strings, and do not mount the Docker socket or writable host data into the server.

The entrypoint preserves arbitrary commands (for example, `python -m sglang.launch_server --help`). Such commands intentionally bypass the preset launcher's authentication and remote-code policy, so configure equivalent controls yourself. The image's default command prints SGLang help; invoking the entrypoint itself with an empty argument list prints the preset usage message.

Offline validation (no Docker or GPU):

```bash
bash tests/test_gpu_selection.sh
python tests/test_secure_launch.py
```

