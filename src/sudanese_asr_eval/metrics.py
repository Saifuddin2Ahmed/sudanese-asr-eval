"""Corpus-level WER/CER with utterance-level bootstrap confidence intervals.

WER = sum(word edits) / sum(reference words) over the whole set (not a mean of
per-utterance WERs). CER is the same over characters, with spaces kept as characters
(jiwer's default). Utterances whose (normalised) reference is empty are excluded from
scoring and counted separately.

Bootstrap: resample utterances with replacement B times (default 1000, fixed seed),
recompute corpus-level WER each time, report the 2.5 / 97.5 percentiles.
Paired bootstrap for system A vs B uses the same resampled indices for both systems.
"""
from __future__ import annotations

import numpy as np
import jiwer

from .textnorm import normalize, raw, whisper_basic

VIEWS = (("raw", raw), ("basic", whisper_basic), ("norm", normalize))


def _edits(refs, hyps, unit):
    errs, lens = [], []
    for r, h in zip(refs, hyps):
        if unit == "word":
            o = jiwer.process_words(r, h if h.strip() else "")
            n = len(r.split())
        else:
            o = jiwer.process_characters(r, h)
            n = len(r)
        errs.append(o.substitutions + o.deletions + o.insertions)
        lens.append(n)
    return np.asarray(errs, dtype=np.int64), np.asarray(lens, dtype=np.int64)


def per_utt(refs, hyps, views=VIEWS):
    """Return dict view -> {'idx','w_err','w_len','c_err','c_len'} arrays (one entry per scored utterance)."""
    out = {}
    for view, f in views:
        r = [f(x) for x in refs]
        h = [f(x) for x in hyps]
        keep = [i for i, x in enumerate(r) if x]
        r = [r[i] for i in keep]
        h = [h[i] for i in keep]
        we, wl = _edits(r, h, "word")
        ce, cl = _edits(r, h, "char")
        out[view] = dict(idx=np.asarray(keep), w_err=we, w_len=wl, c_err=ce, c_len=cl)
    return out


def bootstrap_ci(err, length, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(err)
    idx = rng.integers(0, n, size=(B, n))
    stats = err[idx].sum(1) / np.maximum(length[idx].sum(1), 1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarize(refs, hyps, B=1000, seed=0):
    pu = per_utt(refs, hyps)
    res = {"n_utts": len(refs)}
    for view, d in pu.items():
        for unit in ("w", "c"):
            e, l = d[f"{unit}_err"], d[f"{unit}_len"]
            point = float(e.sum() / max(l.sum(), 1))
            lo, hi = bootstrap_ci(e, l, B, seed)
            name = ("wer" if unit == "w" else "cer") + "_" + view
            res[name] = round(100 * point, 2)
            res[name + "_ci95"] = [round(100 * lo, 2), round(100 * hi, 2)]
        res[f"n_scored_{view}"] = int(len(d["idx"]))
    return res


def paired_bootstrap(refs, hyps_a, hyps_b, view="norm", unit="w", B=1000, seed=0):
    """CI of WER(A) - WER(B) on the same utterances, plus P(A >= B) under resampling."""
    a, b = per_utt(refs, hyps_a)[view], per_utt(refs, hyps_b)[view]
    ea, eb, l = a[f"{unit}_err"], b[f"{unit}_err"], a[f"{unit}_len"]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(l), size=(B, len(l)))
    L = np.maximum(l[idx].sum(1), 1)
    diff = (ea[idx].sum(1) - eb[idx].sum(1)) / L
    return dict(delta=round(100 * float((ea.sum() - eb.sum()) / l.sum()), 2),
                ci95=[round(100 * float(np.percentile(diff, 2.5)), 2), round(100 * float(np.percentile(diff, 97.5)), 2)],
                p_a_not_better=float((diff >= 0).mean()))




def corpus_rate(err, length):
    """Corpus-level error rate in % from per-utterance edit counts and lengths."""
    err, length = np.asarray(err), np.asarray(length)
    return 100.0 * float(err.sum() / max(length.sum(), 1))


def bootstrap_ci_counts(err, length, B=1000, seed=42):
    """95 % bootstrap CI (in %) from per-utterance counts, same resampling as `summarize`."""
    lo, hi = bootstrap_ci(np.asarray(err, dtype=np.int64), np.asarray(length, dtype=np.int64), B, seed)
    return 100 * lo, 100 * hi


def paired_bootstrap_counts(err_a, err_b, length, B=1000, seed=42):
    """Paired bootstrap of rate(A) - rate(B) from per-utterance counts (same as `paired_bootstrap`)."""
    ea, eb, l = (np.asarray(x, dtype=np.int64) for x in (err_a, err_b, length))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(l), size=(B, len(l)))
    L = np.maximum(l[idx].sum(1), 1)
    diff = (ea[idx].sum(1) - eb[idx].sum(1)) / L
    return dict(delta=round(100 * float((ea.sum() - eb.sum()) / l.sum()), 2),
                ci95=[round(100 * float(np.percentile(diff, 2.5)), 2), round(100 * float(np.percentile(diff, 97.5)), 2)],
                p_a_not_better=float((diff >= 0).mean()))
