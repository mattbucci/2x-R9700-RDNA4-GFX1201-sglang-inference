#!/opt/conda/envs/sglang-rdna4/bin/python
"""Launch SGLang with file-backed credentials that never enter process argv."""

from __future__ import annotations

import os
import stat
import sys
from typing import NoReturn

_MAX_SECRET_BYTES = 512


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"ERROR: {message}")


def _read_secret_file(env_name: str) -> str:
    path = os.environ.pop(env_name, "")
    if not path:
        _fail(f"{env_name} is required")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        _fail(f"{env_name} cannot be opened safely: {exc.strerror}")

    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            _fail(f"{env_name} must reference a regular file")
        if file_stat.st_mode & 0o022:
            _fail(f"{env_name} must not be writable by group or other")
        raw = os.read(fd, _MAX_SECRET_BYTES + 2)
    finally:
        os.close(fd)

    if raw.endswith(b"\n"):
        raw = raw[:-1]
    if not raw or b"\n" in raw or b"\r" in raw:
        _fail(f"{env_name} must contain exactly one non-empty line")
    if len(raw) < 32 or len(raw) > _MAX_SECRET_BYTES:
        _fail(f"{env_name} must contain between 32 and {_MAX_SECRET_BYTES} bytes")
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError:
        _fail(f"{env_name} must contain ASCII characters only")
    if any(ord(char) < 33 or ord(char) > 126 for char in value):
        _fail(f"{env_name} must not contain whitespace or control characters")
    return value


def _resolved_value(server_args, name):
    """Read a field as resolution sees it (v0.5.20 records keep declarations in
    a stash and never write the field; older releases write in place)."""
    resolved_dict = getattr(server_args, "resolved_dict", None)
    if callable(resolved_dict):
        try:
            return resolved_dict()[name]
        except (KeyError, TypeError):
            pass
    return getattr(server_args, name)


def _resolve_server_args(server_args, **fields):
    """Set policy/credential fields on the parsed ServerArgs.

    SGLang v0.5.17+ makes the resolved ServerArgs read-only (plain attribute
    assignment raises AttributeError once the declarations are materialized).
    The sanctioned pre-publish mutation point is, per release:
    ``sglang.srt.arg_groups.overrides.declare_resolution(server_args, source,
    **fields)`` (v0.5.20: the record is a msgspec Struct; declarations go to a
    stash that is projected into the config bags at publish and travel with
    the pickled record to every child process, so ``resolved_dict()`` -- not
    the field -- shows the value), ``_late_resolution(source, **fields)``
    (v0.5.18) / ``override(source, **fields)`` (v0.5.17), which write in place.
    Older releases have neither and accept plain assignment.
    """
    try:
        from sglang.srt.arg_groups.overrides import declare_resolution
    except ImportError:
        declare_resolution = None
    if declare_resolution is not None:
        declare_resolution(server_args, "secure-launch", **fields)
        for name, value in fields.items():
            if _resolved_value(server_args, name) != value:
                _fail(f"secure launcher could not set server_args.{name}")
        return
    for method in ("_late_resolution", "override"):
        resolver = getattr(server_args, method, None)
        if callable(resolver):
            try:
                resolver("secure-launch", **fields)
            except TypeError:
                continue
            for name, value in fields.items():
                if getattr(server_args, name) != value:
                    _fail(f"secure launcher could not set server_args.{name}")
            return
    for name, value in fields.items():
        setattr(server_args, name, value)


def main() -> None:
    if os.environ.get("SGLANG_API_KEY") or os.environ.get("SGLANG_ADMIN_API_KEY"):
        _fail("direct credential environment variables are disabled")

    protected = (
        "--api-key",
        "--admin-api-key",
        "--config",
        "--trust-remote-code",
        "--enable-metrics",
        "--max-queued-requests",
        "--enable-custom-logit-processor",
        "--kv-events-config",
        "--enable-forward-pass-metrics",
        "--forward-pass-metrics-ipc-name",
        "--forward-pass-metrics-worker-id",
        "--grpc-mode",
        "--encoder-only",
        "--language-only",
        "--encoder-urls",
        "--encoder-bootstrap-port",
        "--encoder-register-urls",
        "--use-ray",
        "--enable-igw",
        "--tokenizer-worker-num",
        "--enable-lora",
        "--lora-paths",
        "--disaggregation-mode",
        "--remote-instance-weight-loader-start-seed-via-transfer-engine",
        "--remote-instance-weight-loader-seed-instance-ip",
        "--remote-instance-weight-loader-seed-instance-service-port",
        "--remote-instance-weight-loader-send-weights-group-ports",
        "--remote-instance-weight-loader-backend",
        "--engine-info-bootstrap-port",
        "--modelexpress-config",
        "--tool-server",
        "--enable-elastic-expert-backup",
        "--elastic-ep-backend",
        "--moe-a2a-backend",
        "--speculative-moe-a2a-backend",
        "--pipeline-parallel-size",
        "--pp-size",
        "--data-parallel-size",
        "--dp-size",
        "--nnodes",
        "--node-rank",
        "--dist-init-addr",
        "--nccl-init-addr",
        "--nccl-port",
    )
    if any(
        option == name or (option.startswith("--") and name.startswith(option))
        for arg in sys.argv[1:]
        for option in (arg.split("=", 1)[0],)
        for name in protected
    ):
        _fail("a protected server option was passed on the command line")

    api_key = _read_secret_file("SGLANG_API_KEY_FILE")
    admin_api_key = _read_secret_file("SGLANG_ADMIN_API_KEY_FILE")
    os.environ.pop("SGLANG_API_KEY", None)
    os.environ.pop("SGLANG_ADMIN_API_KEY", None)
    if api_key == admin_api_key:
        _fail("API and admin credentials must be distinct")

    trust_remote_code = os.environ.pop("SGLANG_TRUST_REMOTE_CODE", "0")
    enable_metrics = os.environ.pop("SGLANG_ENABLE_METRICS", "0")
    max_queued_requests = os.environ.pop("SGLANG_MAX_QUEUED_REQUESTS", "32")
    if trust_remote_code not in {"0", "1"}:
        _fail("SGLANG_TRUST_REMOTE_CODE must be 0 or 1")
    if enable_metrics not in {"0", "1"}:
        _fail("SGLANG_ENABLE_METRICS must be 0 or 1")
    if not max_queued_requests.isdigit() or int(max_queued_requests) < 1:
        _fail("SGLANG_MAX_QUEUED_REQUESTS must be a positive integer")
    os.environ["SGLANG_ENABLE_GRPC"] = "0"
    os.environ.pop("SGLANG_GRPC_PORT", None)
    os.environ.pop("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE", None)
    os.environ.pop("MASTER_ADDR", None)
    os.environ.pop("MASTER_PORT", None)
    os.environ.pop("NCCL_COMM_ID", None)
    for name in tuple(os.environ):
        if name.startswith("DUMPER_") or name.startswith(
            "SGLANG_TEST_SCRIPTED_RUNTIME"
        ):
            os.environ.pop(name, None)
    os.environ["SGLANG_USE_PICKLE_IPC"] = "0"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    os.environ["NCCL_IB_DISABLE"] = "1"

    # Import only after credentials have been validated and removed from the
    # environment. They are assigned directly to ServerArgs in process memory.
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    parse_argv = list(sys.argv[1:])
    if trust_remote_code == "1":
        parse_argv.append("--trust-remote-code")
    if enable_metrics == "1":
        parse_argv.append("--enable-metrics")
    parse_argv.extend(["--max-queued-requests", max_queued_requests])
    server_args = prepare_server_args(parse_argv)

    # Enforce policy on the parsed object as well as the shell command. This
    # closes argparse abbreviations and config-file indirection.
    unsafe_modes = {
        "grpc_mode": getattr(server_args, "grpc_mode", False),
        "enable_grpc": getattr(server_args, "enable_grpc", False),
        "encoder_only": getattr(server_args, "encoder_only", False),
        "language_only": getattr(server_args, "language_only", False),
        "encoder_urls": bool(getattr(server_args, "encoder_urls", None)),
        "encoder_register_urls": bool(
            getattr(server_args, "encoder_register_urls", None)
        ),
        "use_ray": getattr(server_args, "use_ray", False),
        "enable_igw": getattr(server_args, "enable_igw", False),
        "tokenizer_worker_num": getattr(server_args, "tokenizer_worker_num", 1) != 1,
        "enable_lora": getattr(server_args, "enable_lora", False),
        "lora_paths": bool(getattr(server_args, "lora_paths", None)),
        "disaggregation_mode": getattr(server_args, "disaggregation_mode", "null")
        != "null",
        "kv_event_publisher": bool(getattr(server_args, "kv_events_config", None)),
        "forward_pass_metrics": getattr(
            server_args, "enable_forward_pass_metrics", False
        ),
        "forward_pass_metrics_endpoint": bool(
            getattr(server_args, "forward_pass_metrics_ipc_name", None)
        ),
        "remote_instance_loader": getattr(server_args, "load_format", None)
        == "remote_instance",
        "speculative_remote_instance_loader": getattr(
            server_args, "speculative_draft_load_format", None
        )
        == "remote_instance",
        "modelexpress": bool(getattr(server_args, "modelexpress_config", None)),
        "remote_weight_seed_service": getattr(
            server_args,
            "remote_instance_weight_loader_start_seed_via_transfer_engine",
            False,
        ),
        "tool_server": bool(getattr(server_args, "tool_server", None)),
        "elastic_expert_backup": getattr(
            server_args, "enable_elastic_expert_backup", False
        ),
        "elastic_ep": getattr(server_args, "elastic_ep_backend", None)
        not in (None, "none"),
        "moe_a2a": getattr(server_args, "moe_a2a_backend", "none") != "none",
        "speculative_moe_a2a": getattr(
            server_args, "speculative_moe_a2a_backend", None
        )
        not in (None, "none"),
        "pipeline_parallel": getattr(server_args, "pp_size", 1) != 1,
        "data_parallel": getattr(server_args, "dp_size", 1) != 1,
        "multi_node": getattr(server_args, "nnodes", 1) != 1
        or getattr(server_args, "node_rank", 0) != 0,
        "external_dist_init": bool(getattr(server_args, "dist_init_addr", None)),
        "fixed_nccl_port": getattr(server_args, "nccl_port", None) is not None,
    }
    enabled_unsafe_modes = [name for name, enabled in unsafe_modes.items() if enabled]
    if enabled_unsafe_modes:
        _fail(
            "secure launcher does not support alternate server mode(s): "
            + ", ".join(enabled_unsafe_modes)
        )

    _resolve_server_args(
        server_args,
        api_key=api_key,
        admin_api_key=admin_api_key,
        trust_remote_code=trust_remote_code == "1",
        enable_metrics=enable_metrics == "1",
        max_queued_requests=int(max_queued_requests),
        enable_custom_logit_processor=False,
    )
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
