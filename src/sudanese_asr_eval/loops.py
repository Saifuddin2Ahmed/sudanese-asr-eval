"""Post-hoc diagnostic: collapse runaway repetition loops in a hypothesis.

A run of 3 or more consecutive copies of the same 1-4-word n-gram is collapsed
to one copy, repeatedly, until nothing changes. The rule is reference-free, but
it was chosen after looking at the outputs, so scores computed with it are
exploratory and are not part of the evaluation protocol. It can also collapse
genuine repetitions such as "لا لا لا".
"""
from __future__ import annotations


def collapse_loops(text: str, max_n: int = 4, min_repeats: int = 3) -> str:
    w = text.split()
    changed = True
    while changed:
        changed = False
        for n in range(1, max_n + 1):
            i, out = 0, []
            while i < len(w):
                g = w[i:i + n]
                k = 1
                while w[i + k * n:i + (k + 1) * n] == g and len(g) == n:
                    k += 1
                if k >= min_repeats:
                    out += g
                    i += k * n
                    changed = True
                else:
                    out.append(w[i])
                    i += 1
            w = out
    return " ".join(w)


def is_hallucination(ref: str, hyp: str) -> bool:
    """The paper's flag: hypothesis longer than 3x the reference and more than 5 words."""
    rl, hl = len(ref.split()), len(hyp.split())
    return hl > 3 * rl and hl > 5
