# FP8 SWE-bench bake-off

\`fp8_bakeoff_matrix.sh\` runs SWE-bench Lite across opencode, little-coder, and claw-code against a local SGLang endpoint. Rollouts run on the host; official per-instance Docker images score the resulting patches.

## Scaffold configuration

- **opencode:** provider \`sglang\`, model \`sweep\`, base URL \`http://127.0.0.1:23334/v1\`.
- **little-coder:** set \`LLAMACPP_BASE_URL=http://127.0.0.1:23334/v1\` and \`LLAMACPP_API_KEY=noop\`; use \`--print\`.
- **claw-code:** set \`OPENAI_BASE_URL\` and \`OPENAI_API_KEY\`; use model \`openai/sweep\` and \`--output-format text\`.

The rollout harness uses \`stdin=DEVNULL\`. Do not request opencode JSON output for long multi-turn sessions. Repository diffs are collected from Git, not agent stdout.

Use \`--shard K/N\` to distribute instances into separate prediction files.

## Scoring

\`score_docker.py\` invokes the official SWE-bench evaluation image for each instance and writes \`scores.jsonl\`. \`score_local.py\` is a compatibility fallback, not the canonical score.

Docker images require substantial storage. Put Docker’s data root on the data disk:

\`\`\`json
{"data-root": "/data/docker"}
\`\`\`

After changing \`/etc/docker/daemon.json\`:

\`\`\`bash
sudo systemctl restart docker
docker info | grep 'Docker Root Dir'
\`\`\`

Prune stopped containers regularly. Remove cached evaluation images only when storage pressure justifies the later re-download:

\`\`\`bash
docker container prune -f
docker image prune -af --filter until=24h
\`\`\`

Do not score while the inference server is active if Docker work would contend for RAM, disk, or PCIe bandwidth. Finish rollouts, stop the server, then score.
