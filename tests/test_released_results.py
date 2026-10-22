"""The released SDN-test hypotheses must re-score to the numbers in results/metrics/."""
import json
import os
import unittest

from sudanese_asr_eval.metrics import summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {
    "zs_small": ("zeroshot_metrics.json", "openai/whisper-small"),
    "zs_medium": ("zeroshot_metrics.json", "openai/whisper-medium"),
    "zs_large_v2": ("zeroshot_metrics.json", "openai/whisper-large-v2"),
    "zs_large_v3": ("zeroshot_metrics.json", "openai/whisper-large-v3"),
    "zs_turbo": ("zeroshot_metrics.json", "openai/whisper-large-v3-turbo"),
    "ft_small": ("ft_whisper_small_metrics.json", None),
    "lora_turbo": ("lora_whisper_large_v3_turbo_metrics.json", None),
}
KEYS = ("wer_raw", "wer_raw_ci95", "wer_basic", "wer_basic_ci95", "wer_norm", "wer_norm_ci95",
        "cer_raw", "cer_raw_ci95", "cer_norm", "cer_norm_ci95")


class TestReleasedResults(unittest.TestCase):
    def test_sdn_test_rescoring(self):
        for name, (fn, run) in RUNS.items():
            with open(os.path.join(ROOT, "results", "hyps_sdn_test", name + ".jsonl"), encoding="utf-8") as f:
                d = [json.loads(line) for line in f]
            with open(os.path.join(ROOT, "results", "metrics", fn), encoding="utf-8") as f:
                m = json.load(f)
            want = (m["runs"][run] if run else m)["sdn_test"]
            got = summarize([x["ref"] for x in d], [x["hyp"] for x in d], B=1000, seed=42)
            for k in KEYS:
                self.assertEqual(got[k], want[k], f"{name} {k}")


if __name__ == "__main__":
    unittest.main()
