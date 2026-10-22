"""Text normalisation for Sudanese Arabic ASR scoring.

Three views are reported for every system:

  raw(s)    Unicode NFC + whitespace collapse. Nothing else.

  basic(s)  Re-implementation of Whisper's ``BasicTextNormalizer()`` (default
            arguments), as used by the Hugging Face Whisper fine-tuning template
            (``do_normalize_eval=True``): lower-case, drop bracketed spans, replace
            every Unicode Mark / Symbol / Punctuation character with a space,
            collapse whitespace. It does NOT unify alef / ya / ta marbuta.

  full(s)   The full Arabic normaliser, in this order:
     1. NFKC (folds Arabic presentation forms, e.g. U+FEFB lam-alef ligature)
     2. remove diacritics U+064B-U+065F, superscript alef U+0670,
        Quranic marks U+06D6-U+06ED and tatweel U+0640
     3. alef variants  أ إ آ ٱ  -> ا
     4. alef maqsura   ى -> ي ;  hamza seats  ؤ -> و , ئ -> ي
     5. ta marbuta     ة -> ه
     6. digits: Eastern-Arabic (٠-٩) and Persian (۰-۹) -> ASCII 0-9.
        Numbers are not verbalised: the corpus writes numbers as dialect words
        ("تسعمية تمنية") and no reliable Sudanese number verbaliser exists.
     7. punctuation and symbols (Unicode categories P* and S*, incl. ، ؛ ؟ « ») -> space
     8. Latin letters lower-cased
     9. a stand-alone conjunction "و" is attached to the next word
        (the corpus writes "و تسعمية" and "وتسعمية" interchangeably)
    10. whitespace collapse

Every view is applied identically to references and hypotheses.
In the paper the views are called raw, basic and norm; ``normalize`` is kept as
an alias of ``full`` so that the original experiment code runs unchanged.
"""
from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭـ]")
_ALEF = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}
_YA_HAMZA = {"ى": "ي", "ؤ": "و", "ئ": "ي"}
_TA = {"ة": "ه"}
_DIGITS = {**{chr(0x0660 + i): str(i) for i in range(10)},
           **{chr(0x06F0 + i): str(i) for i in range(10)}}
_TRANS = str.maketrans({**_ALEF, **_YA_HAMZA, **_TA, **_DIGITS})
_WAW = re.compile(r"(?:(?<=\s)|^)و\s+(?=\S)")
_BRACKETS = re.compile(r"[<\[][^>\]]*[>\]]|\(([^)]+?)\)")


def raw(s: str) -> str:
    """Unicode NFC and whitespace collapse only."""
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def whisper_basic(s: str) -> str:
    """Whisper ``BasicTextNormalizer()`` with default arguments (see module doc)."""
    s = (s or "").lower()
    s = _BRACKETS.sub("", s)
    s = "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))
    return " ".join(s.split())


def _strip_punct(s: str) -> str:
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)


def full(s: str) -> str:
    """Full Arabic normaliser (the paper's *norm* view)."""
    s = unicodedata.normalize("NFKC", s or "")
    s = _DIACRITICS.sub("", s)
    s = s.translate(_TRANS)
    s = _strip_punct(s)
    s = s.lower()
    s = " ".join(s.split())
    s = _WAW.sub("و", s)
    return " ".join(s.split())


basic = whisper_basic
normalize = full

#: name -> function, in the order used in all tables
VIEWS = {"raw": raw, "basic": whisper_basic, "norm": full}
ALIASES = {"full": "norm", "whisper_basic": "basic"}


def get_view(name: str):
    name = ALIASES.get(name, name)
    if name not in VIEWS:
        raise KeyError(f"unknown view {name!r}; choose from {sorted(VIEWS) + sorted(ALIASES)}")
    return VIEWS[name]


def ablation_views():
    """Cumulative steps of the full normaliser, as in the paper's ablation table.

    Returns [(label, fn)]: raw, +p/d (NFKC, diacritics, punctuation, lower-case),
    +alef, +ya (ya and hamza seats), +ta (ta marbuta), +wa (= norm).
    Digit mapping is applied with the last step only; references contain no digits.
    """
    t_alef, t_ya, t_ta = (str.maketrans(m) for m in (_ALEF, _YA_HAMZA, _TA))

    def stage(k):
        def f(s):
            s = unicodedata.normalize("NFKC", s or "")
            s = _DIACRITICS.sub("", s)
            s = " ".join(_strip_punct(s).lower().split())
            if k >= 1:
                s = s.translate(t_alef)
            if k >= 2:
                s = s.translate(t_ya)
            if k >= 3:
                s = s.translate(t_ta)
            if k >= 4:
                s = " ".join(_WAW.sub("و", s).split())
            return s
        return f

    return [("raw", raw), ("+p/d", stage(0)), ("+alef", stage(1)), ("+ya", stage(2)),
            ("+ta", stage(3)), ("+wa", full)]
