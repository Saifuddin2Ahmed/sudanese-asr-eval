import unittest

from sudanese_asr_eval.metrics import (bootstrap_ci_counts, paired_bootstrap, paired_bootstrap_counts, per_utt,
                                       summarize)

REFS = ["في يوم الأربعاء", "تعبان خالص", "الكبيرة الماكنة بي كم", "و الله"]
HYPS = ["فى يوم الاربعاء", "تعبان", "الكبيره الماكنه بكم", "والله"]


class TestMetrics(unittest.TestCase):
    def test_summarize(self):
        s = summarize(REFS, HYPS, B=200, seed=42)
        self.assertLess(s["wer_norm"], s["wer_raw"])
        self.assertEqual(s["n_utts"], 4)
        lo, hi = s["wer_raw_ci95"]
        self.assertLessEqual(lo, s["wer_raw"])
        self.assertGreaterEqual(hi, s["wer_raw"])
        # corpus-level WER, not a mean of per-utterance WERs:
        # raw edits 2 + 1 + 4 + 2 = 9 over 3 + 2 + 4 + 2 = 11 reference words
        self.assertAlmostEqual(s["wer_raw"], round(100 * 9 / 11, 2))

    def test_perfect(self):
        s = summarize(REFS, REFS, B=50)
        self.assertEqual(s["wer_raw"], 0.0)
        self.assertEqual(s["cer_norm"], 0.0)

    def test_empty_reference_is_dropped(self):
        s = summarize(["", "زول"], ["كلام", "زول"], B=20)
        self.assertEqual(s["n_scored_raw"], 1)
        self.assertEqual(s["wer_raw"], 0.0)

    def test_counts_api_matches_text_api(self):
        pu = per_utt(REFS, HYPS)["norm"]
        s = summarize(REFS, HYPS, B=300, seed=7)
        lo, hi = bootstrap_ci_counts(pu["w_err"], pu["w_len"], B=300, seed=7)
        self.assertEqual([round(lo, 2), round(hi, 2)], s["wer_norm_ci95"])
        pb = per_utt(REFS, REFS)["norm"]
        a = paired_bootstrap(REFS, HYPS, REFS, view="norm", B=300, seed=7)
        b = paired_bootstrap_counts(pu["w_err"], pb["w_err"], pu["w_len"], B=300, seed=7)
        self.assertEqual(a, b)
        self.assertGreater(a["delta"], 0)
        self.assertEqual(a["p_a_not_better"], 1.0)


if __name__ == "__main__":
    unittest.main()
