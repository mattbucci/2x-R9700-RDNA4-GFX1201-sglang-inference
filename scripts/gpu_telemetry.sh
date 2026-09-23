#!/usr/bin/env bash
# gpu_telemetry.sh — cheap per-card telemetry to JSONL, for post-mortems.
#
# Added after 2026-09-22, when GPU 0000:07:00.0 dropped off the PCIe bus mid-decode
# with nothing logged anywhere: no temperature, power, clock or link history existed
# for the minutes before the loss. This samples sysfs/hwmon (no GPU compute, no
# rocm-smi) every INTERVAL seconds and appends one JSON line per card to
# $OUT_DIR/gpu-YYYY-MM-DD.jsonl. A card in runtime suspend is recorded as
# {"state":"suspended"} without touching hwmon (a hwmon read would resume it).
# A card whose PCI config space reads 0xffff while runtime-active is recorded as
# {"state":"lost"} — the timestamp of the first such sample bounds the loss.
#
#   nohup setsid scripts/gpu_telemetry.sh > /data/logs/gpu-telemetry/logger.log 2>&1 < /dev/null &
#   echo $! > /data/logs/gpu-telemetry/logger.pid
#
# Env: INTERVAL (s, default 30), OUT_DIR (default /data/logs/gpu-telemetry),
#      MIN_VRAM_MIB (default 16384: skip the iGPU).
set -uo pipefail
INTERVAL=${INTERVAL:-30}
OUT_DIR=${OUT_DIR:-/data/logs/gpu-telemetry}
MIN_VRAM_MIB=${MIN_VRAM_MIB:-16384}
mkdir -p "$OUT_DIR"

rd() { local v; v=$(cat "$1" 2>/dev/null) || v=""; printf '%s' "$v"; }
num() { local v; v=$(rd "$1"); [[ "$v" =~ ^-?[0-9]+$ ]] && printf '%s' "$v" || printf 'null'; }

cards=()
for d in /sys/bus/pci/devices/*; do
  [ "$(rd "$d/vendor")" = "0x1002" ] || continue
  [ -f "$d/mem_info_vram_total" ] || continue
  tot=$(( $(num "$d/mem_info_vram_total") / 1048576 ))
  [ "$tot" -ge "$MIN_VRAM_MIB" ] && cards+=("$(basename "$d")")
done
echo "[telemetry $(date '+%F %T')] cards: ${cards[*]} interval=${INTERVAL}s out=$OUT_DIR"

while true; do
  ts=$(date '+%Y-%m-%dT%H:%M:%S%z')
  f="$OUT_DIR/gpu-$(date +%F).jsonl"
  for pci in "${cards[@]}"; do
    d=/sys/bus/pci/devices/$pci
    rs=$(rd "$d/power/runtime_status")
    if [ "$rs" != "active" ]; then
      printf '{"ts":"%s","pci":"%s","state":"suspended","runtime_status":"%s"}\n' "$ts" "$pci" "$rs" >> "$f"
      continue
    fi
    cfg=$(head -c 2 "$d/config" 2>/dev/null | od -An -tx1 | tr -d ' ')
    if [ "$cfg" != "0210" ]; then
      printf '{"ts":"%s","pci":"%s","state":"lost","config_bytes":"%s","link":"%s x%s"}\n' \
        "$ts" "$pci" "$cfg" "$(rd "$d/current_link_speed")" "$(rd "$d/current_link_width")" >> "$f"
      continue
    fi
    hw=$(ls -d "$d"/hwmon/hwmon* 2>/dev/null | head -n 1)
    # temps are millidegrees; power microwatts; freq Hz; in0 mV
    t_edge=$(num "$hw/temp1_input"); t_junc=$(num "$hw/temp2_input"); t_mem=$(num "$hw/temp3_input")
    p_avg=$(num "$hw/power1_average"); p_cap=$(num "$hw/power1_cap")
    fan=$(num "$hw/fan1_input"); sclk=$(num "$hw/freq1_input"); mclk=$(num "$hw/freq2_input"); vdd=$(num "$hw/in0_input")
    busy=$(num "$d/gpu_busy_percent"); mbusy=$(num "$d/mem_busy_percent")
    vram=$(num "$d/mem_info_vram_used")
    printf '{"ts":"%s","pci":"%s","state":"active","edge_c":%s,"junction_c":%s,"mem_c":%s,"power_w":%s,"power_cap_w":%s,"fan_rpm":%s,"sclk_mhz":%s,"mclk_mhz":%s,"vddgfx_mv":%s,"gpu_busy":%s,"mem_busy":%s,"vram_used_mib":%s,"link":"%s x%s","perf_level":"%s"}\n' \
      "$ts" "$pci" \
      "$(python3 -c "import sys;v=sys.argv[1];print('null' if v=='null' else int(v)/1000)" "$t_edge")" \
      "$(python3 -c "import sys;v=sys.argv[1];print('null' if v=='null' else int(v)/1000)" "$t_junc")" \
      "$(python3 -c "import sys;v=sys.argv[1];print('null' if v=='null' else int(v)/1000)" "$t_mem")" \
      "$(python3 -c "import sys;v=sys.argv[1];print('null' if v=='null' else int(v)/1e6)" "$p_avg")" \
      "$(python3 -c "import sys;v=sys.argv[1];print('null' if v=='null' else int(v)/1e6)" "$p_cap")" \
      "$fan" \
      "$(python3 -c "import sys;v=sys.argv[1];print('null' if v=='null' else int(v)//1000000)" "$sclk")" \
      "$(python3 -c "import sys;v=sys.argv[1];print('null' if v=='null' else int(v)//1000000)" "$mclk")" \
      "$vdd" "$busy" "$mbusy" \
      "$(python3 -c "import sys;v=sys.argv[1];print('null' if v=='null' else int(v)//1048576)" "$vram")" \
      "$(rd "$d/current_link_speed")" "$(rd "$d/current_link_width")" "$(rd "$d/power_dpm_force_performance_level")" >> "$f"
  done
  sleep "$INTERVAL"
done
