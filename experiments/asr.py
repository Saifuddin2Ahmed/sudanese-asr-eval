"""Whisper loading + batched greedy transcription (shared by zero-shot and fine-tuning)."""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

GEN_MAX_NEW_TOKENS = 225  # same cap as Mansour's runs (generation_max_length=225)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _local_if_bin_only(model_id: str) -> str:
    """Repos with only pytorch_model.bin make transformers start a background
    'auto_conversion' thread that contacts the Hub's safetensors-convert service (it can
    open a PR on someone else's repo, and it delays exit by minutes). Loading the same
    files from a local snapshot avoids both. Repos with safetensors load directly."""
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import list_repo_files, snapshot_download
    files = list_repo_files(model_id)
    if any(f.endswith(".safetensors") for f in files):
        return model_id
    return snapshot_download(model_id, allow_patterns=["*.json", "pytorch_model*.bin", "*.txt"])


def load(model_id: str, gen_config_from: str | None = None, dtype=None, device=None):
    device, auto_dtype = device_dtype() if device is None else (device, dtype)
    dtype = dtype or auto_dtype
    proc = WhisperProcessor.from_pretrained(gen_config_from or model_id)
    model = WhisperForConditionalGeneration.from_pretrained(_local_if_bin_only(model_id), dtype=dtype).to(device).eval()
    if gen_config_from:  # old 2022 checkpoints lack lang_to_id in generation_config
        model.generation_config = GenerationConfig.from_pretrained(gen_config_from)
    return model, proc


@torch.no_grad()
def transcribe(model, proc, samples, batch_size=16, language="ar", num_beams=1, log_every=20):
    """Greedy (num_beams=1) decoding, language forced to Arabic unless language=None.
    Clips are sorted by length for batching; output order matches `samples`."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    order = sorted(range(len(samples)), key=lambda i: len(samples[i]["audio"]))
    hyps = [None] * len(samples)
    t0 = time.time()
    for b, start in enumerate(range(0, len(order), batch_size)):
        idx = order[start:start + batch_size]
        feats = proc.feature_extractor([samples[i]["audio"] for i in idx], sampling_rate=16000,
                                       return_tensors="pt").input_features.to(device, dtype)
        kw = dict(task="transcribe", max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=num_beams)
        if language:
            kw["language"] = language
        out = model.generate(feats, **kw)
        for i, text in zip(idx, proc.batch_decode(out, skip_special_tokens=True)):
            hyps[i] = text.strip()
        if log_every and b % log_every == 0:
            print(f"    batch {b + 1}/{(len(order) + batch_size - 1) // batch_size}  {time.time() - t0:.0f}s", flush=True)
    return hyps


def save_hyps(path, samples, hyps):
    with open(path, "w", encoding="utf-8") as f:
        for s, h in zip(samples, hyps):
            f.write(json.dumps(dict(id=s["id"], ref=s["text"], hyp=h), ensure_ascii=False) + "\n")


def env_info():
    import platform
    import transformers
    info = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    try:
        import peft
        info["peft"] = peft.__version__
    except ImportError:
        pass
    try:
        import jiwer
        info["jiwer"] = getattr(jiwer, "__version__", "?")
    except ImportError:
        pass
    return info


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p
