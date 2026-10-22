"""Turn hypothesis files into text-free per-clip error counts.

    python scripts/export_per_clip.py HYPS_DIR [--out results/per_clip]

HYPS_DIR holds the `hyps__<run>__<set>.jsonl` files written by the experiment runners
(keys id, ref, hyp). For every system and test set this writes one TSV with, per clip:

  id, ref_words, hyp_words          raw whitespace word counts (hallucination flag)
  <view>_w_err, <view>_w_len,       word edits and reference words
  <view>_c_err, <view>_c_len        character edits and reference characters

for the views raw, basic, norm, the cumulative ablation steps (abl_pd, abl_alef,
abl_ya, abl_ta; the last step equals norm) and the post-hoc loop-collapsed
hypotheses (loop_raw, loop_norm). A clip whose reference is empty in a view has
empty cells for that view, exactly as `metrics.per_utt` drops it.

The TSVs contain no transcript text, so they can be shared for test sets whose text
may not be redistributed (OOOK-Eval). Clip order is the order of the hyps file, which
is what the bootstrap resampling depends on.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

from sudanese_asr_eval.loops import collapse_loops
from sudanese_asr_eval.metrics import _edits
from sudanese_asr_eval.textnorm import ablation_views, full, raw, whisper_basic

#: run name inside the file name -> short system key used in all tables
SYSTEMS = {
    "openai__whisper-small": "zs_small",
    "openai__whisper-medium": "zs_medium",
    "openai__whisper-large-v2": "zs_large_v2",
    "openai__whisper-large-v2@auto-lang": "zs_large_v2_auto",
    "openai__whisper-large-v3": "zs_large_v3",
    "openai__whisper-large-v3-turbo": "zs_turbo",
    "AymanMansour__Whisper-Sudanese-Dialect-small (2022, external ref)": "mansour2022_small",
    "ft_whisper_small": "ft_small",
    "lora_whisper_large_v3_turbo": "lora_turbo",
}
_ABL = dict(ablation_views())
VIEWS = [("raw", raw, False), ("basic", whisper_basic, False), ("norm", full, False),
         ("abl_pd", _ABL["+p/d"], False), ("abl_alef", _ABL["+alef"], False),
         ("abl_ya", _ABL["+ya"], False), ("abl_ta", _ABL["+ta"], False),
         ("loop_raw", raw, True), ("loop_norm", full, True)]


def export(path, out_dir):
    m = re.match(r"hyps__(.+)__(sdn_test|ookeval)\.jsonl$", os.path.basename(path))
    if not m or m.group(1) not in SYSTEMS:
        return None
    system, test_set = SYSTEMS[m.group(1)], m.group(2)
    d = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    cols = {"id": [x["id"] for x in d],
            "ref_words": [len(raw(x["ref"]).split()) for x in d],
            "hyp_words": [len(raw(x["hyp"]).split()) for x in d]}
    for name, f, loop in VIEWS:
        refs = [f(x["ref"]) for x in d]
        hyps = [f(collapse_loops(x["hyp"]) if loop else x["hyp"]) for x in d]
        keep = [i for i, r in enumerate(refs) if r]
        we, wl = _edits([refs[i] for i in keep], [hyps[i] for i in keep], "word")
        ce, cl = _edits([refs[i] for i in keep], [hyps[i] for i in keep], "char")
        for key, arr in (("w_err", we), ("w_len", wl), ("c_err", ce), ("c_len", cl)):
            col = [""] * len(d)
            for j, i in enumerate(keep):
                col[i] = int(arr[j])
            cols[f"{name}_{key}"] = col
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{system}__{test_set}.tsv")
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        w.writerows(zip(*cols.values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hyps_dir")
    ap.add_argument("--out", default=os.path.join("results", "per_clip"))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.hyps_dir, "**", "hyps__*.jsonl"), recursive=True))
    for p in files:
        o = export(p, a.out)
        print(("wrote " + o) if o else ("skipped " + p))


if __name__ == "__main__":
    main()
