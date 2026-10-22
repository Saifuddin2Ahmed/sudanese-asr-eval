"""Evaluation tools for Sudanese Arabic ASR: text normalisation (raw / basic / full)
and corpus-level WER/CER with bootstrap confidence intervals and paired tests."""
from .textnorm import VIEWS, basic, full, get_view, normalize, raw, whisper_basic

__version__ = "1.0.0"
__all__ = ["VIEWS", "basic", "full", "get_view", "normalize", "raw", "whisper_basic", "__version__"]
