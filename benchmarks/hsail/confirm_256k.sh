#!/bin/bash
# 256K-confirmation sweep (batches: 3090 cross-team ask 31B+devstral + our gemma SWA-ratio validation).
# Run at bake-off pause/boundary. Per model: serve, read achieved KV-pool max-total + single-seq 256K decode point, probes.
# Receipts → benchmarks/hsail/confirm_256k_results.md
set -u
R=~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference
OUT=$R/benchmarks/hsail/confirm_256k_results.md
echo "# 256K confirmation $(date)" >> $OUT
stop(){ pkill -f "[s]glang.launch_server"; sleep 8; pkill -9 -f "[s]glang::scheduler" 2>/dev/null; sleep 4; }

# model | preset | extra
for row in \
  "gemma4-31b|gemma4-31b|--mem-fraction-static 0.92" \
  "devstral|devstral|--mem-fraction-static 0.92" \
  "gemma4|gemma4|" ; do
  IFS='|' read -r label preset extra <<< "$row"
  stop
  echo "== $label ($preset) $(date +%H:%M) ==" | tee -a $OUT
  setsid bash -c "EXTRA_ARGS=\"$extra\" $R/scripts/launch.sh $preset --port 23334 --context-length 262144 > /tmp/exp-awq-decode/serve-256k-$label.log 2>&1 & disown" </dev/null & disown
  for _ in $(seq 1 140); do curl -sf -m5 http://127.0.0.1:23334/health >/dev/null 2>&1 && break; grep -qiE "OutOfMemory|core dumped|Error" /tmp/exp-awq-decode/serve-256k-$label.log && break; sleep 10; done
  # achieved KV pool
  grep -oE "max_total_num_tokens=[0-9]+|#token: [0-9]+|KV cache.*[0-9]+ tokens|swa-full-tokens" /tmp/exp-awq-decode/serve-256k-$label.log | tail -3 | tee -a $OUT
  curl -s http://127.0.0.1:23334/get_model_info 2>/dev/null | head -c 200 | tee -a $OUT; echo | tee -a $OUT
  # 256K single-seq decode point
  python3 - "$label" <<'PY' | tee -a $OUT
import json,sys,time,urllib.request
lbl=sys.argv[1]; words="lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
prompt=(words*40000)[:255000*5]
b=json.dumps({"model":"default","prompt":prompt,"max_tokens":64,"temperature":0.7,"ignore_eos":True}).encode()
try:
    t0=time.time(); r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:23334/v1/completions",b,{"Content-Type":"application/json"}),timeout=900)
    d=json.load(r); print(f"{lbl}: prompt_tokens={d['usage']['prompt_tokens']} decode64 in {time.time()-t0:.0f}s -> SERVES 256K" if d['usage']['prompt_tokens']>200000 else f"{lbl}: only {d['usage']['prompt_tokens']} tok (capped below 256K)")
except Exception as e:
    print(f"{lbl}: 256K FAIL {type(e).__name__}: {str(e)[:120]}")
PY
done
stop
echo "DONE 256K confirmation" | tee -a $OUT
