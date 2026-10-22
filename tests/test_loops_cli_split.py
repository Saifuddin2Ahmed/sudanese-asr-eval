import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

from sudanese_asr_eval.loops import collapse_loops, is_hallucination

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestLoops(unittest.TestCase):
    def test_collapse(self):
        self.assertEqual(collapse_loops("و زول زول زول زول"), "و زول")
        self.assertEqual(collapse_loops("أسمع كدي كدي كدي كدي"), "أسمع كدي")
        self.assertEqual(collapse_loops("يا زول يا زول يا زول تعال"), "يا زول تعال")
        self.assertEqual(collapse_loops("لا لا"), "لا لا")  # two copies are kept

    def test_flag(self):
        self.assertTrue(is_hallucination("زول", "زول زول زول زول زول زول"))
        self.assertFalse(is_hallucination("زول", "زول زول زول"))


def run_cli(*args, stdin=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, "-m", "sudanese_asr_eval", *args], input=stdin, capture_output=True,
                       env=env, encoding="utf-8", check=True)
    return p.stdout


class TestCLI(unittest.TestCase):
    def test_normalize_stdin(self):
        self.assertEqual(run_cli("normalize", stdin="ألف و تسعمية\n").strip(), "الف وتسعميه")
        out = run_cli("normalize", "--view", "all", stdin="أهلاً، مدرسة\n").splitlines()
        self.assertEqual(out[0].split("\t"), ["input", "raw", "basic", "norm"])
        self.assertEqual(out[1].split("\t")[1:], ["أهلاً، مدرسة", "أهلا مدرسة", "اهلا مدرسه"])

    def test_score_and_compare(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.jsonl"), os.path.join(d, "b.jsonl")
            rows = [("1", "في يوم الأربعاء", "فى يوم الاربعاء"), ("2", "تعبان خالص", "تعبان")]
            with open(a, "w", encoding="utf-8") as f:
                for i, r, h in rows:
                    f.write(json.dumps(dict(id=i, ref=r, hyp=h), ensure_ascii=False) + "\n")
            with open(b, "w", encoding="utf-8") as f:
                for i, r, _ in rows:
                    f.write(json.dumps(dict(id=i, ref=r, hyp=r), ensure_ascii=False) + "\n")
            s = json.loads(run_cli("score", a, "--bootstrap", "50"))
            self.assertIn("wer_norm_ci95", s)
            c = json.loads(run_cli("compare", a, b, "--view", "raw", "--bootstrap", "50"))
            self.assertEqual(c["n_clips"], 2)
            self.assertGreater(c["delta"], 0)


class TestSplit(unittest.TestCase):
    def test_split_file(self):
        path = os.path.join(ROOT, "splits", "sdn_split_v1.tsv")
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        self.assertNotIn("\r", text)
        lines = text.splitlines()
        rows = list(csv.DictReader([x for x in lines if not x.startswith("#")], delimiter="\t"))
        self.assertEqual(list(rows[0]), ["id", "split", "speaker", "gender", "program", "dur"])
        n = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")}
        self.assertEqual(n, {"train": 2695, "dev": 302, "test": 369})
        spk = {s: {r["speaker"] for r in rows if r["split"] == s} for s in n}
        self.assertFalse(spk["train"] & spk["dev"] or spk["train"] & spk["test"] or spk["dev"] & spk["test"])
        self.assertEqual(sorted(spk["test"], key=int), ["6", "10", "17", "20", "26"])
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        self.assertTrue(all(float(r["dur"]) <= 30 for r in rows))


if __name__ == "__main__":
    unittest.main()
