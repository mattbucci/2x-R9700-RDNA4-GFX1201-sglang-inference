set -u
REPO=/home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference; W=/data/sgl-v0522-replay; LIVE=/data/sgl-v0522
git -C $W reset -q --hard baa9bef && git -C $W clean -qfdx
n=0; fail=0
for p in $(ls $REPO/patches/[0-9][0-9][0-9]-*.patch | sort); do
  n=$((n+1))
  git -C $W apply --check "$p" 2>/dev/null && git -C $W apply --index "$p" 2>/dev/null || { echo "APPLY FAIL: $p"; fail=$((fail+1)); }
done
echo "patches=$n apply_failures=$fail"
echo "--- byte/mode equivalence (every represented path)"
bad=0; cnt=0
while IFS= read -r -d '' f; do
  cnt=$((cnt+1))
  if [ -L "$W/$f" ]; then
    [ "$(readlink "$W/$f")" = "$(readlink "$LIVE/$f" 2>/dev/null)" ] || { echo "SYMLINK DIFF: $f"; bad=$((bad+1)); }
  elif [ -e "$W/$f" ]; then
    [ -e "$LIVE/$f" ] || { echo "MISSING in live: $f"; bad=$((bad+1)); continue; }
    cmp -s "$W/$f" "$LIVE/$f" || { echo "CONTENT DIFF: $f"; bad=$((bad+1)); }
    [ "$(stat -c %a "$W/$f")" = "$(stat -c %a "$LIVE/$f")" ] || { echo "MODE DIFF: $f"; bad=$((bad+1)); }
  else
    [ -e "$LIVE/$f" ] && { echo "SHOULD BE DELETED in live: $f"; bad=$((bad+1)); }
  fi
done < <(git -C $W diff --cached --name-only -z)
echo "paths=$cnt mismatches=$bad"
echo "--- git diff --check"; git -C $W diff --cached --check && echo "diff --check clean"
echo "--- second apply must be rejected"
ok2=0; for p in $(ls $REPO/patches/[0-9][0-9][0-9]-*.patch | sort); do git -C $W apply --check "$p" >/dev/null 2>&1 && { echo "SECOND APPLY ACCEPTED: $p"; ok2=$((ok2+1)); }; done; echo "second_apply_accepted=$ok2"
