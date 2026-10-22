"""Command-line interface.

  sudanese-asr-eval normalize [--view raw|basic|norm|all] [FILE]   (stdin if no FILE)
  sudanese-asr-eval score HYPS.jsonl [--bootstrap 1000] [--seed 42]
  sudanese-asr-eval compare A.jsonl B.jsonl [--view norm] [--unit w|c]

HYPS files are JSON Lines with at least the keys "id", "ref" and "hyp"
(the format written by experiments/asr.py). `compare` pairs clips by id.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .textnorm import VIEWS, get_view


def _utf8_stdio():
    for s in (sys.stdin, sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cmd_normalize(a):
    src = open(a.file, encoding="utf-8") if a.file else sys.stdin
    views = list(VIEWS) if a.view == "all" else [a.view]
    fns = [get_view(v) for v in views]
    if a.view == "all":
        print("\t".join(["input"] + views))
    for line in src:
        line = line.rstrip("\r\n")
        out = [f(line) for f in fns]
        print("\t".join([line] + out) if a.view == "all" else out[0])


def cmd_score(a):
    from .metrics import summarize
    d = _read_jsonl(a.hyps)
    res = summarize([x["ref"] for x in d], [x["hyp"] for x in d], B=a.bootstrap, seed=a.seed)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_compare(a):
    from .metrics import paired_bootstrap
    A = {x["id"]: x for x in _read_jsonl(a.a)}
    B = {x["id"]: x for x in _read_jsonl(a.b)}
    ids = [i for i in A if i in B]
    if not ids:
        sys.exit("no common clip ids")
    if any(A[i]["ref"] != B[i]["ref"] for i in ids):
        sys.exit("references differ between the two files")
    r = paired_bootstrap([A[i]["ref"] for i in ids], [A[i]["hyp"] for i in ids], [B[i]["hyp"] for i in ids],
                         view=get_view_name(a.view), unit=a.unit, B=a.bootstrap, seed=a.seed)
    r.update(n_clips=len(ids), view=get_view_name(a.view), unit=a.unit)
    print(json.dumps(r, indent=1))


def get_view_name(v):
    get_view(v)  # validates
    return {"full": "norm", "whisper_basic": "basic"}.get(v, v)


def main(argv=None):
    _utf8_stdio()
    ap = argparse.ArgumentParser(prog="sudanese-asr-eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("normalize", help="normalise text, one line at a time")
    n.add_argument("file", nargs="?")
    n.add_argument("--view", default="norm", choices=["raw", "basic", "norm", "full", "all"])
    n.set_defaults(func=cmd_normalize)
    s = sub.add_parser("score", help="corpus WER/CER in all views with bootstrap CIs")
    s.add_argument("hyps")
    s.add_argument("--bootstrap", type=int, default=1000)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_score)
    c = sub.add_parser("compare", help="paired bootstrap: rate(A) - rate(B) on the same clips")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--view", default="norm", choices=["raw", "basic", "norm", "full"])
    c.add_argument("--unit", default="w", choices=["w", "c"])
    c.add_argument("--bootstrap", type=int, default=1000)
    c.add_argument("--seed", type=int, default=42)
    c.set_defaults(func=cmd_compare)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
