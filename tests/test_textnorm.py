import unittest

from sudanese_asr_eval import full, get_view, raw, whisper_basic
from sudanese_asr_eval.textnorm import ablation_views

CASES = {
    "فِي يوم الأربعاء، أول ديسمبر!": "في يوم الاربعاء اول ديسمبر",
    "ألف و تسعمية تمنية و أربعين": "الف وتسعميه تمنيه واربعين",
    "مدرسة ٣ و ٤": "مدرسه 3 و4",
    "مســـاء الخير؟": "مساء الخير",
    "إلى مسؤول": "الي مسوول",
    "OK يا Ahmed.": "ok يا ahmed",
    "و": "و",
    "": "",
}


class TestFull(unittest.TestCase):
    def test_cases(self):
        for src, want in CASES.items():
            self.assertEqual(full(src), want, src)

    def test_idempotent(self):
        for src in CASES:
            self.assertEqual(full(full(src)), full(src))

    def test_none_safe(self):
        self.assertEqual(full(None), "")
        self.assertEqual(raw(None), "")
        self.assertEqual(whisper_basic(None), "")

    def test_ablation_last_step_is_full(self):
        steps = ablation_views()
        self.assertEqual([label for label, _ in steps], ["raw", "+p/d", "+alef", "+ya", "+ta", "+wa"])
        for src in CASES:
            self.assertEqual(steps[-1][1](src), full(src))


class TestRawBasic(unittest.TestCase):
    def test_raw_only_whitespace(self):
        self.assertEqual(raw("  أهلاً   وسهلاً،\tيا  زول "), "أهلاً وسهلاً، يا زول")

    def test_basic_keeps_letter_variants(self):
        # basic removes marks and punctuation but does NOT unify alef or ta marbuta
        self.assertEqual(whisper_basic("أهلاً، مدرسة!"), "أهلا مدرسة")

    def test_views(self):
        self.assertIs(get_view("norm"), full)
        self.assertIs(get_view("full"), full)
        self.assertIs(get_view("basic"), whisper_basic)
        with self.assertRaises(KeyError):
            get_view("nope")

    def test_basic_matches_transformers(self):
        try:
            from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        except Exception:
            self.skipTest("transformers not installed")
        ref = BasicTextNormalizer()
        for src in list(CASES) + ["فِي (مرحبا) [x] يوم، «الخير»"]:
            self.assertEqual(whisper_basic(src).split(), ref(src).split(), src)


if __name__ == "__main__":
    unittest.main()
