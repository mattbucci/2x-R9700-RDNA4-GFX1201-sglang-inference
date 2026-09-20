"""Offline checks for workdirs.py: slug determinism, both layouts resolve, pi slugs."""
import unittest

from workdirs import (NAMED_ROOTS, NEUTRAL_ROOT, Resolver, container_dir, is_instance_dir,
                      is_work_dir, neutral_slug, pi_cwd_slug, pi_store_globs, top_dir)


class WorkdirsTest(unittest.TestCase):
    def test_slug_is_opaque_and_stable(self):
        s = neutral_slug("django__django-11564")
        self.assertEqual(s, neutral_slug("django__django-11564"))
        self.assertRegex(s, r"^repo-[0-9a-f]{10}$")
        self.assertNotIn("django", s)
        self.assertNotIn("11564", s)
        self.assertNotEqual(s, neutral_slug("django__django-11565"))

    def test_container_dir(self):
        self.assertEqual(container_dir("a__b-1", False), "/data/swebench-work/a__b-1")
        self.assertEqual(container_dir("a__b-1", False, NAMED_ROOTS[1]), "/tmp/swebench-work/a__b-1")
        self.assertEqual(container_dir("a__b-1", True), NEUTRAL_ROOT + neutral_slug("a__b-1"))

    def test_resolver_both_layouts(self):
        r = Resolver(["django__django-11564", "astropy__astropy-6938"])
        self.assertEqual(r.instance_of("/data/swebench-work/django__django-11564"), "django__django-11564")
        self.assertEqual(r.instance_of("/data/swebench-work/django__django-11564/tests"), "django__django-11564")
        self.assertEqual(r.instance_of("/tmp/swebench-work/astropy__astropy-6938"), "astropy__astropy-6938")
        self.assertEqual(r.instance_of(container_dir("astropy__astropy-6938", True)), "astropy__astropy-6938")
        self.assertEqual(r.instance_of(container_dir("astropy__astropy-6938", True) + "/astropy"), "astropy__astropy-6938")
        self.assertIsNone(r.instance_of("/work/repo-0000000000"))  # unknown slug
        self.assertIsNone(r.instance_of("/home/u/project"))
        self.assertIsNone(r.instance_of("/work/other"))
        self.assertIsNone(r.instance_of(""))
        self.assertTrue(r.matches("/data/swebench-work/django__django-11564/x", "django__django-11564"))
        self.assertFalse(r.matches("/data/swebench-work/django__django-11564", "django__django-11565"))
        # named layout never depended on the known set
        self.assertEqual(Resolver().instance_of("/data/swebench-work/x__y-1"), "x__y-1")

    def test_is_instance_dir(self):
        iid = "django__django-11564"
        self.assertTrue(is_instance_dir("/data/swebench-work/" + iid, iid))
        self.assertTrue(is_instance_dir("/tmp/swebench-work/" + iid + "/django/db", iid))
        self.assertTrue(is_instance_dir(container_dir(iid, True), iid))
        self.assertTrue(is_instance_dir(container_dir(iid, True) + "/tests", iid))
        self.assertFalse(is_instance_dir("/data/swebench-work/" + iid + "5", iid))
        self.assertFalse(is_instance_dir(container_dir(iid + "5", True), iid))
        self.assertFalse(is_instance_dir("", iid))

    def test_is_work_dir_and_top(self):
        self.assertTrue(is_work_dir("/data/swebench-work/x__y-1"))
        self.assertTrue(is_work_dir("/work/repo-abcdef0123"))
        self.assertFalse(is_work_dir("/data/swebench-work/"))
        self.assertFalse(is_work_dir("/work/"))
        self.assertFalse(is_work_dir("/work/notarepo"))
        self.assertEqual(top_dir("/work/repo-abcdef0123/sub"), "/work/repo-abcdef0123")
        self.assertEqual(top_dir("/data/swebench-work/x__y-1/sub/dir"), "/data/swebench-work/x__y-1")

    def test_pi_slugs(self):
        self.assertEqual(pi_cwd_slug("/data/swebench-work/a__b-1"), "--data-swebench-work-a__b-1--")
        self.assertIn("--data-swebench-work-*--", pi_store_globs())
        self.assertIn("--work-repo-*--", pi_store_globs())


if __name__ == "__main__":
    unittest.main()
