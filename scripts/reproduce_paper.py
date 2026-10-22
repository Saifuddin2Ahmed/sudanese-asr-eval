"""Recompute every number in the paper from the released per-clip counts (CPU, about a minute).

    python scripts/reproduce_paper.py [--per_clip results/per_clip] [--check]

Prints and writes results/paper_numbers.json:
  main      WER/CER per system and test set, raw / basic / norm, with 95 % bootstrap CIs
            (Table 1 of the paper; Tables 1a and 1b of the results notes)
  gain      raw - norm WER, the normalisation effect discussed in Section 4
  paired    paired bootstrap deltas (A - B) on the same clips
  ablation  cumulative normalisation steps (Appendix B)
  halluc    clips with hyp > 3x ref and > 5 words, and their share of all word errors
  loops     exploratory loop-collapse scores and paired deltas (post hoc)
  speakers  per-speaker norm WER on the SDN test set (Limitations)

With --check it also compares the main table with the metrics JSONs written on
Kaggle (results/metrics/) and exits with status 1 on any difference.
Bootstrap: 1,000 resamples, seed 42, exactly as in the original runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

from sudanese_asr_eval.metrics import bootstrap_ci_counts, corpus_rate, paired_bootstrap_counts

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
B, SEED = 1000, 42
SETS = ("ookeval", "sdn_test")
ORDER = ["zs_small", "zs_medium", "zs_large_v2", "zs_large_v2_auto", "zs_large_v3", "zs_turbo",
         "mansour2022_small", "ft_small", "lora_turbo"]
PAIRS = [("lora_turbo", "zs_turbo"), ("lora_turbo", "zs_large_v3"), ("lora_turbo", "ft_small"),
         ("ft_small", "zs_small"), ("ft_small", "mansour2022_small"), ("lora_turbo", "mansour2022_small"),
         ("zs_large_v3", "mansour2022_small"), ("zs_large_v2", "zs_large_v2_auto"), ("zs_large_v3", "zs_turbo")]
LOOP_SYSTEMS = ["zs_large_v3", "zs_turbo", "mansour2022_small", "ft_small", "lora_turbo"]
LOOP_PAIRS = {"ookeval": [("lora_turbo", "zs_large_v3"), ("lora_turbo", "zs_turbo"), ("lora_turbo", "mansour2022_small"),
                          ("zs_large_v3", "mansour2022_small"), ("lora_turbo", "ft_small")],
              "sdn_test": [("lora_turbo", "zs_large_v3"), ("lora_turbo", "zs_turbo"), ("lora_turbo", "ft_small")]}
ABLATION = [("raw", "raw"), ("+p/d", "abl_pd"), ("+alef", "abl_alef"), ("+ya", "abl_ya"), ("+ta", "abl_ta"), ("+wa", "norm")]
METRICS_SRC = {
    "zs_small": ("zeroshot_metrics.json", "openai/whisper-small"),
    "zs_medium": ("zeroshot_metrics.json", "openai/whisper-medium"),
    "zs_large_v2": ("zeroshot_metrics.json", "openai/whisper-large-v2"),
    "zs_large_v2_auto": ("zeroshot_metrics.json", "openai/whisper-large-v2@auto-lang"),
    "zs_large_v3": ("zeroshot_metrics.json", "openai/whisper-large-v3"),
    "zs_turbo": ("zeroshot_metrics.json", "openai/whisper-large-v3-turbo"),
    "mansour2022_small": ("zeroshot_metrics.json", "AymanMansour/Whisper-Sudanese-Dialect-small (2022, external ref)"),
    "ft_small": ("ft_whisper_small_metrics.json", None),
    "lora_turbo": ("lora_whisper_large_v3_turbo_metrics.json", None),
}


def load(per_clip):
    data = {}
    for s in SETS:
        for name in ORDER:
            p = os.path.join(per_clip, f"{name}__{s}.tsv")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    data[(name, s)] = list(csv.DictReader(f, delimiter="\t"))
    return data


def counts(rows, view, unit):
    """(err, len) arrays over the clips scored in this view, in file order."""
    e, l = [], []
    for r in rows:
        if r[f"{view}_{unit}_len"] != "":
            e.append(int(r[f"{view}_{unit}_err"]))
            l.append(int(r[f"{view}_{unit}_len"]))
    return np.array(e, dtype=np.int64), np.array(l, dtype=np.int64)


def rate_ci(rows, view, unit):
    e, l = counts(rows, view, unit)
    lo, hi = bootstrap_ci_counts(e, l, B, SEED)
    return round(corpus_rate(e, l), 2), [round(lo, 2), round(hi, 2)]


def paired(ra, rb, view, unit="w"):
    ids_b = {r["id"]: r for r in rb}
    ra = [r for r in ra if r["id"] in ids_b and r[f"{view}_{unit}_len"] != ""]
    ea = [int(r[f"{view}_{unit}_err"]) for r in ra]
    eb = [int(ids_b[r["id"]][f"{view}_{unit}_err"]) for r in ra]
    ln = [int(r[f"{view}_{unit}_len"]) for r in ra]
    return paired_bootstrap_counts(ea, eb, ln, B, SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_clip", default=os.path.join(ROOT, "results", "per_clip"))
    ap.add_argument("--metrics", default=os.path.join(ROOT, "results", "metrics"))
    ap.add_argument("--split", default=os.path.join(ROOT, "splits", "sdn_split_v1.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "paper_numbers.json"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    data = load(a.per_clip)
    if not data:
        sys.exit(f"no per-clip files in {a.per_clip}")
    out = defaultdict(dict)

    print("== Main table: WER/CER % [95 % CI]")
    for s in SETS:
        for n in ORDER:
            if (n, s) not in data:
                continue
            rows = data[(n, s)]
            m = {"n_utts": len(rows)}
            for view in ("raw", "basic", "norm"):
                for unit, label in (("w", "wer"), ("c", "cer")):
                    m[f"{label}_{view}"], m[f"{label}_{view}_ci95"] = rate_ci(rows, view, unit)
            out["main"][f"{n}|{s}"] = m
            out["gain"][f"{n}|{s}"] = round(m["wer_raw"] - m["wer_norm"], 2)
            print(f"{s:8s} {n:18s} WER raw {m['wer_raw']:6.2f} {m['wer_raw_ci95']}  basic {m['wer_basic']:6.2f}  "
                  f"norm {m['wer_norm']:6.2f} {m['wer_norm_ci95']}  CER raw {m['cer_raw']:6.2f}  norm {m['cer_norm']:6.2f}"
                  f"  (raw-norm {out['gain'][f'{n}|{s}']:+.2f})")

    print("\n== Paired bootstrap, WER(A) - WER(B) [95 % CI]")
    for s in SETS:
        for x, y in PAIRS:
            if (x, s) in data and (y, s) in data:
                for view in ("raw", "norm"):
                    r = paired(data[(x, s)], data[(y, s)], view)
                    out["paired"][f"{x} - {y}|{s}|{view}"] = r
                print(f"{s:8s} {x} - {y}: raw {out['paired'][f'{x} - {y}|{s}|raw']['delta']:+.2f} "
                      f"{out['paired'][f'{x} - {y}|{s}|raw']['ci95']}  norm {out['paired'][f'{x} - {y}|{s}|norm']['delta']:+.2f} "
                      f"{out['paired'][f'{x} - {y}|{s}|norm']['ci95']}")

    print("\n== Normalisation ablation (cumulative WER %): " + " | ".join(l for l, _ in ABLATION))
    for s in SETS:
        for n in LOOP_SYSTEMS:
            if (n, s) in data:
                row = [round(corpus_rate(*counts(data[(n, s)], v, "w")), 2) for _, v in ABLATION]
                out["ablation"][f"{n}|{s}"] = dict(zip([l for l, _ in ABLATION], row))
                print(f"{s:8s} {n:18s} " + " ".join(f"{v:6.2f}" for v in row))

    print("\n== Hallucinated clips (hyp > 3x ref and > 5 words) and their share of word errors")
    for s in SETS:
        for n in ORDER:
            if (n, s) not in data:
                continue
            rows = data[(n, s)]
            flag = [int(r["hyp_words"]) > 3 * int(r["ref_words"]) and int(r["hyp_words"]) > 5 for r in rows]
            share = {}
            for view in ("raw", "norm"):
                tot = sum(int(r[f"{view}_w_err"]) for r in rows if r[f"{view}_w_len"] != "")
                hal = sum(int(r[f"{view}_w_err"]) for r, f in zip(rows, flag) if f and r[f"{view}_w_len"] != "")
                share[view] = round(100 * hal / max(tot, 1), 1)
            out["halluc"][f"{n}|{s}"] = dict(clips=int(sum(flag)), of=len(rows), error_share_pct=share)
            print(f"{s:8s} {n:18s} {sum(flag):3d} / {len(rows)}   share of word errors: raw {share['raw']}%, norm {share['norm']}%")

    print("\n== Loop collapse (exploratory, post hoc)")
    for s in SETS:
        for n in LOOP_SYSTEMS:
            if (n, s) in data:
                rows = data[(n, s)]
                r_raw, ci_raw = rate_ci(rows, "loop_raw", "w")
                r_norm, ci_norm = rate_ci(rows, "loop_norm", "w")
                out["loops"][f"{n}|{s}"] = dict(wer_raw=r_raw, wer_raw_ci95=ci_raw, wer_norm=r_norm, wer_norm_ci95=ci_norm)
                print(f"{s:8s} {n:18s} raw {r_raw:6.2f} {ci_raw}  norm {r_norm:6.2f} {ci_norm}")
        for x, y in LOOP_PAIRS[s]:
            if (x, s) in data and (y, s) in data:
                for view in ("raw", "norm"):
                    r = paired(data[(x, s)], data[(y, s)], f"loop_{view}")
                    out["loops"][f"{x} - {y}|{s}|{view}"] = r
                    print(f"{s:8s}   paired {x} - {y} {view}: {r['delta']:+.2f} {r['ci95']}")

    print("\n== Per-speaker norm WER on the SDN test set")
    spk = {}
    with open(a.split, encoding="utf-8") as f:
        for r in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"):
            spk[r["id"]] = f"{r['speaker']}/{r['gender'][0]}"
    for n in ORDER:
        if (n, "sdn_test") not in data:
            continue
        g = defaultdict(lambda: [0, 0])
        for r in data[(n, "sdn_test")]:
            if r["norm_w_len"] != "":
                g[spk[r["id"]]][0] += int(r["norm_w_err"])
                g[spk[r["id"]]][1] += int(r["norm_w_len"])
        out["speakers"][n] = {k: [round(100 * e / l, 1), l] for k, (e, l) in sorted(g.items())}
        print(f"{n:18s} " + "  ".join(f"spk {k}: {v[0]}% ({v[1]} w)" for k, v in out["speakers"][n].items()))

    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nwrote", a.out)

    if a.check:
        bad = 0
        for key, m in out["main"].items():
            n, s = key.split("|")
            fn, run = METRICS_SRC[n]
            p = os.path.join(a.metrics, fn)
            ref = json.load(open(p, encoding="utf-8"))
            ref = (ref["runs"][run] if run else ref)[s]
            for k, v in m.items():
                if k in ref and ref[k] != v:
                    bad += 1
                    print(f"MISMATCH {key} {k}: recomputed {v} vs Kaggle {ref[k]}")
        print("check:", "OK, every main-table number matches the Kaggle metrics JSONs" if not bad else f"{bad} mismatches")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
