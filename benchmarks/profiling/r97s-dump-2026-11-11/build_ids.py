"""R97-S: token ids of the dump prompts (CPU only, no server).

usage: build_ids.py <repo> <commit> <tokenizer dir> <out.json> <tail> <depth> [<depth> ...]
The prompt is a chat (system + one user message + generation prompt) whose user message holds this repository's
Python sources at <commit> (git show, sorted paths) followed by a question. For each depth P the id list has
exactly P + <tail> ids: the template head, source ids cut to fit, then the question, the end of the user turn and the
generation prompt. The cold request sends ids[:P], the turn and the flushed repeat send all P + <tail>.
"""
import hashlib
import json
import os
import subprocess
import sys

from transformers import AutoTokenizer

MARK = "@@R97S-SOURCES@@"
SYSTEM = "You are a careful code reviewer."
QUESTION = "\n\nSummarize what the files above do, then list the three most likely bugs with file and function names."


def main():
    repo, commit, tok_dir, out_path, tail = sys.argv[1:6]
    tail = int(tail)
    depths = [int(x) for x in sys.argv[6:]]
    tok = AutoTokenizer.from_pretrained(tok_dir)
    sha = subprocess.run(["git", "-C", repo, "rev-parse", commit], capture_output=True, text=True, check=True).stdout.strip()
    paths = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", sha, "scripts"],
                           capture_output=True, text=True, check=True).stdout.split("\n")
    paths = sorted(p for p in paths if p.endswith(".py"))
    need = max(depths) + tail
    src, used, chars = [], [], 0
    for p in paths:
        body = subprocess.run(["git", "-C", repo, "show", "%s:%s" % (sha, p)], capture_output=True, text=True, check=True).stdout
        src.append("# ---- file: %s\n%s\n" % (p, body))
        used.append(p)
        chars += len(src[-1])
        if chars > need * 6:  # far more characters than any tokenizer needs for `need` tokens of code
            break
    src_ids = tok("".join(src), add_special_tokens=False)["input_ids"]
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": MARK + QUESTION}],
        tokenize=False, add_generation_prompt=True)
    assert text.count(MARK) == 1, text.count(MARK)
    pre, post = text.split(MARK)
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    post_ids = tok(post, add_special_tokens=False)["input_ids"]
    out = {"commit": sha, "tokenizer": tok_dir, "tail": tail, "files_available": len(paths), "files_read": len(used),
           "source_ids": len(src_ids), "template_head_ids": len(pre_ids), "template_tail_ids": len(post_ids),
           "template_head_text": pre, "template_tail_text": post, "prompts": {}}
    for depth in depths:
        room = depth + tail - len(pre_ids) - len(post_ids)
        assert 0 < room <= len(src_ids), (depth, room, len(src_ids))
        ids = pre_ids + src_ids[:room] + post_ids
        assert len(ids) == depth + tail
        out["prompts"][str(depth)] = {
            "prefix": depth, "ids": ids, "source_ids_used": room,
            "sha256": hashlib.sha256((",".join(map(str, ids))).encode()).hexdigest(),
            "tail_text": tok.decode(ids[depth:]),
        }
    with open(out_path + ".tmp", "w") as f:
        json.dump(out, f)
        f.write("\n")
    os.replace(out_path + ".tmp", out_path)
    print("commit", sha[:12], "files", len(used), "of", len(paths), "source ids", len(src_ids),
          "head", len(pre_ids), "tail", len(post_ids))
    print("head text", repr(pre[:200]))
    print("tail text", repr(post[:300]))
    for depth in depths:
        p = out["prompts"][str(depth)]
        print("depth", depth, "ids", len(p["ids"]), "sha", p["sha256"][:16], "first", p["ids"][:6], "last", p["ids"][-6:])
        print("   tail rows decode:", repr(p["tail_text"][-260:]))


if __name__ == "__main__":
    main()
