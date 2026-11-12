#!/usr/bin/env python3
"""consol_capture_boot_logs.sh against a stub server: what is kept per preset, what reaches the launcher, a preset
that does not boot, done marks. The launcher is only asked for its print-only line; nothing here opens a GPU."""
import os
import subprocess
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DRIVER = os.path.join(REPO, "scripts", "bench", "consol_capture_boot_logs.sh")
FIRED = "The server is fired up and ready to roll!"
LAZY = "lazily logged line of the first request"

STUB_LAUNCH = """#!/bin/bash
# stands in for scripts/launch.sh <preset>: a log with a boot part and a later line, then a process that stays
echo "$1" >> "$STUB_OUT/launches.txt"
env > "$STUB_OUT/launch-env-$1.txt"
if [ "${STUB_FAIL:-}" = "$1" ]; then echo "RuntimeError: stub boot failure"; exit 1; fi
[ "${STUB_DEVTREE:-}" = "$1" ] && export PYTHONPATH=/data/sgl-v0522-stub/python
echo "boot line of $1"; echo "%s"; echo "%s"
echo $$ > "$STUB_OUT/stub.pid"; touch "$STUB_OUT/stub.up"
exec sleep 300
""" % (FIRED, LAZY)

STUB_PROBE = """#!/bin/bash
# stands in for the decode requests: writes the receipt named by --out
while [ $# -gt 0 ]; do [ "$1" = --out ] && echo '{}' > "$2"; shift; done
"""


def presets_on_this_host(names):
    for n in names:
        line = subprocess.run(["bash", "scripts/launch.sh", n], cwd=REPO, capture_output=True, text=True,
                              env=dict(os.environ, LAUNCH_DRY_RUN="1")).stdout
        parts = line.split("--model-path ")
        if len(parts) < 2 or not os.path.isdir(parts[1].split()[0]):
            return False
    return True


PRESETS = ["laguna", "north-mini"]
HAVE = presets_on_this_host(PRESETS)


@unittest.skipUnless(HAVE, "the checkpoints of the test presets are not on this host")
class Capture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "out")
        self.launch = os.path.join(self.tmp.name, "stub_launch.sh")
        self.probe = os.path.join(self.tmp.name, "stub_probe.sh")
        for path, text in ((self.launch, STUB_LAUNCH), (self.probe, STUB_PROBE)):
            with open(path, "w") as f:
                f.write(text)

    def tearDown(self):
        self.tmp.cleanup()

    def run_driver(self, presets, trace=False, **extra):
        env = dict(os.environ, CAPTURE_STUB="1", CAPTURE_OUT=self.out, CAPTURE_STUB_LAUNCH=self.launch,
                   CAPTURE_STUB_PROBE=self.probe, **extra)
        env.pop("CAPTURE_DRY", None)
        r = subprocess.run(["bash"] + (["-x"] if trace else []) + [DRIVER] + presets, cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=300)
        return r, [l for l in r.stdout.splitlines() if "CONSOL BOOTLOG DONE" in l]

    def read(self, name):
        with open(os.path.join(self.out, name)) as f:
            return f.read()

    def test_the_boot_log_is_cut_at_the_fired_up_line_and_the_whole_log_is_kept(self):
        r, done = self.run_driver(["laguna"], SGLANG_STRAY="1", MEM="0.5")
        self.assertEqual(r.returncode, 0, r.stdout[-400:])
        self.assertTrue(done and done[-1].endswith("gate=captured:1/1"), done)
        boot, whole = self.read("boot-laguna.log"), self.read("server-laguna.log")
        self.assertTrue(boot.rstrip().endswith(FIRED))
        self.assertNotIn(LAZY, boot)
        self.assertIn(LAZY, whole)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "DONE.laguna")))
        self.assertEqual(self.read("decode-laguna.json").strip(), "{}")
        self.assertIn("--random-seed 7", self.read("launch-laguna.txt"))

    def test_nothing_of_the_calling_shell_reaches_the_launcher(self):
        self.run_driver(["laguna"], SGLANG_STRAY="1", MEM="0.5", QUANT_OVERRIDE="x", PYTHONPATH="/data/sgl-v0522-dev/python",
                        PORT="9", CTX="2048", LAUNCH_DRY_RUN="0")
        names = {l.split("=", 1)[0]: l.split("=", 1)[1] for l in self.read("launch-env-laguna.txt").splitlines() if "=" in l}
        for gone in ("SGLANG_STRAY", "MEM", "QUANT_OVERRIDE", "PYTHONPATH", "PORT", "CTX", "LAUNCH_DRY_RUN"):
            self.assertNotIn(gone, names)
        self.assertEqual(names.get("EXTRA_ARGS"), "--random-seed 7")
        self.assertNotIn("SGLANG_STRAY", self.read("environ-laguna.txt"))
        self.assertIn("--port 23334", self.read("launch-laguna.txt"))

    def test_a_preset_that_does_not_boot_is_named_and_the_next_one_runs(self):
        r, done = self.run_driver(PRESETS, trace=True, STUB_FAIL="laguna")
        self.assertTrue(done and done[-1].endswith("gate=captured:1/2:failed:laguna"), done)
        self.assertEqual(self.read("launches.txt").split(), ["laguna", "laguna", "north-mini"])
        self.assertFalse(os.path.exists(os.path.join(self.out, "DONE.laguna")))
        self.assertTrue(os.path.isfile(os.path.join(self.out, "boot-north-mini.log")))
        self.assertNotIn("scripts/free_gpu.sh >", r.stderr.replace("for f in scripts/launch.sh scripts/free_gpu.sh", ""))
        self.assertEqual([l for l in r.stderr.splitlines() if l.lstrip("+ ").startswith("bash scripts/free_gpu.sh")], [])

    def test_a_development_tree_on_the_server_path_is_not_a_capture(self):
        r, done = self.run_driver(["laguna"], STUB_DEVTREE="laguna")
        self.assertTrue(done and done[-1].endswith("gate=captured:0/1:failed:laguna"), done)
        self.assertIn("development tree", r.stdout)

    def test_done_marks_skip_a_captured_preset(self):
        self.run_driver(["laguna"])
        r, done = self.run_driver(PRESETS)
        self.assertTrue(done and done[-1].endswith("gate=captured:2/2"), done)
        self.assertIn("laguna: done mark present, skipped", r.stdout)
        self.assertEqual(self.read("launches.txt").split(), ["laguna", "north-mini"])


class Refusals(unittest.TestCase):
    def test_stub_mode_refuses_the_real_log_location(self):
        r = subprocess.run(["bash", DRIVER, "laguna"], cwd=REPO, capture_output=True, text=True,
                           env=dict(os.environ, CAPTURE_STUB="1", CAPTURE_OUT="/data/logs/consol-bootlogs-stub-test"))
        self.assertEqual(r.returncode, 2)
        self.assertIn("outside /data/logs", r.stdout)
        self.assertFalse(os.path.exists("/data/logs/consol-bootlogs-stub-test"))

    def test_a_preset_the_launcher_does_not_know_stops_before_any_boot(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(["bash", DRIVER, "no-such-preset"], cwd=REPO, capture_output=True, text=True,
                               env=dict(os.environ, CAPTURE_DRY="1", CAPTURE_OUT=os.path.join(d, "out")))
        self.assertIn("CONSOL BOOTLOG DONE gate=no-preset:no-such-preset", r.stdout)


if __name__ == "__main__":
    unittest.main()
