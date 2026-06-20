#!/usr/bin/env python
# #32 quality gate: needle-in-haystack on the windowed server (one boot, two needle depths).
# Pure recent-window N=4096 → decode attends only the last ~4096 tokens. Prediction:
#   needle EARLY (~5% depth, far outside window) → FAIL (model can't attend it)
#   needle LATE  (last ~500 tokens, inside window) → PASS
# This characterizes the recall tradeoff (it's a hard architectural limit, not soft degradation).
import sys, json, re, urllib.request, urllib.error
port, mname, ctxfile = sys.argv[1], sys.argv[2], sys.argv[3]
TOKBUDGET = float(sys.argv[4]) if len(sys.argv) > 4 else 230000.0
full = open(ctxfile).read()

NEEDLE_VAL = "ZEPHYR-4419"
NEEDLE = f"\n\n>>> IMPORTANT FACT: The secret passphrase is {NEEDLE_VAL}. Remember it. <<<\n\n"
Q = f"\n\n---\nWhat is the secret passphrase mentioned earlier in this document? Reply with ONLY the passphrase value, nothing else."

def send_raw(ctx):
    req = {"model": mname, "temperature": 0, "max_tokens": 40,
           "messages": [{"role": "user", "content": ctx + Q}]}
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                data=json.dumps(req).encode(),
                                headers={"Content-Type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(rq, timeout=2400).read().decode())
        return d["choices"][0]["message"]["content"], None
    except urllib.error.HTTPError as e:
        try: msg = json.loads(e.read().decode()).get("message","")
        except Exception: msg = str(e)
        return None, msg
    except Exception as e:
        return None, str(e)

# Self-calibrate the char→token ratio: send the full file once, read the actual token count
# from the "is longer (N tokens)" 400, compute chars-per-token, then build slices to TOKBUDGET.
_, msg = send_raw(full)
m = re.search(r"\((\d+) tokens\)", msg or "")
full_tok = int(m.group(1)) if m else 833321
cpt = len(full) / full_tok   # chars per token
keep_chars = int(TOKBUDGET * cpt * 0.95)   # 5% safety under the pool
base = full[:keep_chars]
print(f"[calib] full={full_tok} tok, cpt={cpt:.2f}, target~{TOKBUDGET:.0f} tok → base {len(base)} chars")

def build(depth_frac):
    cut = int(len(base) * depth_frac)
    return base[:cut] + NEEDLE + base[cut:]

for label, frac in [("EARLY(~5%)", 0.05), ("LATE(end)", 0.999)]:
    ans, err = send_raw(build(frac))
    if err:
        print(f"needle {label:12s}: ERR {err[:80]!r}"); continue
    hit = NEEDLE_VAL in ans
    print(f"needle {label:12s}: {'PASS (recalled)' if hit else 'FAIL (not recalled)'} | resp={ans[:70]!r}")
