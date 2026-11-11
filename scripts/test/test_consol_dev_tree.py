"""consol_dev_tree.py on a throwaway repository: the builder never runs against a serving tree here."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
import consol_dev_tree as dev  # noqa: E402
import consol_session as ses  # noqa: E402
import patch_inventory as inv  # noqa: E402

A0 = "import os\n\n\ndef first():\n    return os.sep\n\n\ndef second():\n    return 2\n"
B0 = "from sglang import a\n\n\ndef run():\n    return a.first()\n\n# TUNED_TABLE is only a word in this comment\n"
A1 = A0 + "\n\ndef helper_alpha_one():\n    return 'alpha line of the first patch'\n"
B3 = B0 + "\n\ndef use():\n    return a.helper_alpha_one()\n"
C2 = "TUNED_TABLE = {'rows': 64}\n"
CFG = "python/sglang/configs/N=64,block_shape=[128, 128].json"
A4 = A1.replace("return 'alpha line of the first patch'", "return 'alpha line, edited by the fourth patch'")


def sh(cwd, *a):
    r = subprocess.run(a, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise AssertionError(f"{a}: {r.stderr}")
    return r.stdout


class Tree(unittest.TestCase):
    def setUp(self):
        self.assertTrue(os.path.isfile(dev.PYFLAKES), "pyflakes is part of the check: a missing binary must not skip it")
        self.d = tempfile.mkdtemp(prefix="consol-tree-test-")
        self.addCleanup(shutil.rmtree, self.d, True)
        src, self.patches = os.path.join(self.d, "src"), os.path.join(self.d, "patches")
        os.makedirs(os.path.join(src, "python", "sglang"))
        os.makedirs(os.path.join(src, "rust"))
        os.makedirs(self.patches)
        self.write(src, "python/sglang/__init__.py", "")
        self.write(src, "python/sglang/a.py", A0)
        self.write(src, "python/sglang/b.py", B0)
        self.write(src, "rust/lib.rs", "// core\n")
        sh(src, "git", "init", "-q", ".")
        sh(src, "git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "add", "-A")
        sh(src, "git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "base")
        self.base = sh(src, "git", "rev-parse", "HEAD").strip()
        for num, files in (("001", {"python/sglang/a.py": A1}), ("002", {"python/sglang/c.py": C2, CFG: "{}\n", "python/sglang/k.cu": "// kernel\n"}),
                           ("003", {"python/sglang/b.py": B3}), ("004", {"python/sglang/a.py": A4})):
            for rel, body in files.items():
                self.write(src, rel, body)
            sh(src, "git", "add", "-A")
            with open(os.path.join(self.patches, f"{num}-step.patch"), "w") as f:
                f.write(sh(src, "git", "diff", "--cached"))
            sh(src, "git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", num)
        self.live = os.path.join(self.d, "live")
        shutil.copytree(src, self.live, ignore=shutil.ignore_patterns(".git"))
        self.write(self.live, "python/sglang/ext.so", "built")
        self.write(self.live, "python/sglang/__pycache__/a.cpython-312.pyc", "stale")
        self.scratch = os.path.join(self.d, "scratch")
        sh(self.d, "git", "clone", "-q", src, self.scratch)
        self.write(self.scratch, inv.MARKER, "")

    def write(self, root, rel, body):
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        with open(os.path.join(root, rel), "w") as f:
            f.write(body)

    def build(self, arm, drop, name="tree", scratch=None):
        out = os.path.join(self.d, "trees", name)
        args = types.SimpleNamespace(arm=arm, drop=drop, arms=None, out=out, live=self.live, scratch=scratch or self.scratch,
                                     base=self.base, patches=self.patches, pyflakes=dev.PYFLAKES)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = dev.build(args)
        rec = None
        if os.path.isfile(os.path.join(out, "consol-tree.json")):
            with open(os.path.join(out, "consol-tree.json")) as f:
                rec = json.load(f)
        return rc, buf.getvalue(), out, rec

    def test_a_patch_that_only_adds_a_file_gives_a_clean_tree(self):
        rc, text, out, rec = self.build("002", "002")
        self.assertEqual((rc, rec["verdict"]), (0, "ok"), text)
        self.assertFalse(os.path.exists(os.path.join(out, "python/sglang/c.py")))
        self.assertEqual(rec["tree_check"], {"file": "sglang/c.py", "missing": True})
        self.assertEqual(rec["removed_files"], sorted(["python/sglang/c.py", CFG, "python/sglang/k.cu"]))
        self.assertEqual((rec["dangling"], rec["native_sources"]), ({}, []))   # a comment is no reference, a removed source no build
        self.assertTrue(os.path.isfile(os.path.join(out, "python/sglang/ext.so")))           # built artefacts come along
        self.assertFalse(os.path.exists(os.path.join(out, "python/sglang/__pycache__")))    # stale bytecode does not
        self.assertEqual(os.path.realpath(os.path.join(out, "rust")), os.path.realpath(os.path.join(self.live, "rust")))
        self.assertTrue(os.path.isfile(os.path.join(self.live, "python/sglang/c.py")))       # the live tree is only read
        self.assertEqual(sh(self.scratch, "git", "status", "--porcelain"), "")           # the scratch clone ends at base
        self.assertEqual(sh(self.scratch, "git", "rev-parse", "HEAD").strip(), self.base)

    def test_a_later_patch_that_stops_applying_is_a_failure_and_no_tree(self):
        rc, text, out, rec = self.build("001", "001")
        self.assertEqual(rc, 1)
        self.assertIn("['004']", text)
        self.assertFalse(os.path.exists(out))

    def test_a_reference_left_without_its_definition_asks_for_review(self):
        rc, text, out, rec = self.build("001", "001,004")
        self.assertEqual((rc, rec["verdict"]), (0, "review"), text)
        self.assertEqual(rec["dangling"], {"helper_alpha_one": ["sglang/b.py"]})
        self.assertEqual(rec["changed_files"], ["python/sglang/a.py"])
        self.assertEqual(rec["tree_check"]["file"], "sglang/a.py")
        with open(os.path.join(out, "python/sglang/a.py")) as f:
            self.assertEqual(f.read(), A0)

    def test_a_change_inside_one_file_names_an_added_line(self):
        rc, text, out, rec = self.build("003", "003")
        self.assertEqual((rc, rec["verdict"]), (0, "ok"), text)
        self.assertIn("helper_alpha_one", rec["tree_check"]["absent"])
        with open(os.path.join(out, "python/sglang/b.py")) as f:
            self.assertEqual(f.read(), B0)

    def check(self, arm):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ses.cmd_check({"arms": [arm]}, os.path.join(self.live, "python"))
        return rc, buf.getvalue()

    def test_the_session_check_accepts_the_tree_until_the_live_tree_moves(self):
        rc, text, out, rec = self.build("003", "003")
        arm = {"name": "p003", "patches": ["003"], "tree": out}
        self.assertEqual(self.check(arm), (0, "CHECK ok: 1 arms, 1 tree arms\n"))
        rc, text = self.check({**arm, "patches": ["002"]})                 # a tree built for another group
        self.assertEqual(rc, 1)
        self.assertIn("the tree drops ['003']", text)
        self.write(self.live, "python/sglang/a.py", A4 + "\n\ndef landed_later():\n    return 5\n")   # a patch lands
        rc, text = self.check(arm)
        self.assertEqual(rc, 1)
        self.assertIn("rebuild it: ['sglang/a.py']", text)

    def test_a_tree_under_review_needs_a_recorded_reason(self):
        rc, text, out, rec = self.build("001", "001,004")
        arm = {"name": "p001", "patches": ["001", "004"], "tree": out}
        rc, text = self.check(arm)
        self.assertEqual(rc, 1)
        self.assertIn("receipt reads review", text)
        self.assertEqual(self.check({**arm, "reviewed": "use() is not on the serving path"})[0], 0)

    def test_an_existing_directory_and_an_unmarked_clone_are_refused(self):
        os.makedirs(os.path.join(self.d, "trees", "taken"))
        with self.assertRaises(SystemExit) as e:
            self.build("002", "002", name="taken")
        self.assertIn("exists", str(e.exception))
        with self.assertRaises(SystemExit) as e:
            self.build("002", "002", name="fresh", scratch=self.live)
        self.assertIn("marker", str(e.exception))
        self.assertTrue(os.path.isfile(os.path.join(self.live, "python/sglang/a.py")))

    def test_a_live_tree_that_drifted_from_the_series_is_a_failure(self):
        self.write(self.live, "python/sglang/b.py", B3 + "\n# hand edit\n")
        rc, text, out, rec = self.build("002", "002")
        self.assertEqual((rc, rec["verdict"]), (1, "fail"), text)
        self.assertIn("outside the group", text)


if __name__ == "__main__":
    unittest.main()
