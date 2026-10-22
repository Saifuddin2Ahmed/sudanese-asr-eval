"""Data loading. No `datasets` Audio decoding (it changes between versions); we read
WAV / parquet bytes ourselves with soundfile and resample with librosa (soxr).

Corpora
  * SDN  - Mansour & Mukhtar, Sudanese dialect speech dataset, Zenodo 10.5281/zenodo.6869079
           v1.1, CC BY 4.0. Read from the ORIGINAL Zenodo zip (md5 below). All reported
           numbers use v1.1; the later v1.2 (record 7426918, same concept DOI
           10.5281/zenodo.6721682) is a different archive and is not checked here. The HF mirror
           `arbml/sudanese_dialect_speech` must NOT be used: its transcripts are shifted by
           1-2 rows against the audio (only 85 / 3,547 rows match the Zenodo text files).
  * OOOK - Mansour's evaluation set used in arXiv 2601.06802 (OOOK-Eval), HF
           `AymanMansour/OOOK-Eval-Final` @ e256ab53 (797 clips). Evaluation only; no licence
           is stated on the card, so we never redistribute it.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import urllib.request
import zipfile

import numpy as np
import soundfile as sf

SR = 16000
SDN_URL = "https://zenodo.org/records/6869079/files/Sudanese_dialect_speech_dataset.zip?download=1"
SDN_MD5 = "61effea8ff14daa690bb0e05f9930719"
SDN_FILES = "SDN Dialect Corpus v1.0/Dataset files/"
OOOK_REPO = "AymanMansour/OOOK-Eval-Final"
OOOK_REV = "e256ab538b6efb2e0e9e452b962d44b198662619"
OOOK_FILE = "data/train-00000-of-00001.parquet"


def _to16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return wav


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_sdn_zip(cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "Sudanese_dialect_speech_dataset.zip")
    if not os.path.exists(path):
        print("[data] downloading SDN corpus from Zenodo (1.45 GB) ...", flush=True)
        tmp = path + ".part"
        with urllib.request.urlopen(SDN_URL, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, path)
    got = _md5(path)
    if got != SDN_MD5:
        raise RuntimeError(f"SDN zip md5 mismatch: {got} != {SDN_MD5}")
    return path


class _HttpFile(io.RawIOBase):
    def __init__(self, url):
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                d = urllib.request.urlopen(req, timeout=120).read()
                break
            except Exception:
                if attempt == 4:
                    raise
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def open_remote_sdn_zip() -> zipfile.ZipFile:
    """Read single clips from the Zenodo zip by HTTP range requests (for small eval runs)."""
    return zipfile.ZipFile(io.BufferedReader(_HttpFile(SDN_URL), buffer_size=1 << 20))


def read_split(tsv_path: str):
    lines = [l for l in open(tsv_path, encoding="utf-8") if not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def load_sdn(tsv_path: str, zip_source, splits=("train", "dev", "test"), limit=None):
    """zip_source: a path, or an already-open zipfile.ZipFile (e.g. over HTTP for tests)."""
    z = zip_source if isinstance(zip_source, zipfile.ZipFile) else zipfile.ZipFile(zip_source)
    rows = read_split(tsv_path)
    out = {s: [] for s in splits}
    for r in rows:
        if r["split"] not in out or (limit and len(out[r["split"]]) >= limit):
            continue
        text = z.read(SDN_FILES + r["id"] + ".txt").decode("utf-8-sig").strip()
        wav, sr = sf.read(io.BytesIO(z.read(SDN_FILES + r["id"] + ".wav")), dtype="float32")
        out[r["split"]].append(dict(id=r["id"], text=text, audio=_to16k(wav, sr),
                                    speaker=r["speaker"], gender=r["gender"], program=r["program"]))
    for s in splits:
        print(f"[data] SDN {s}: {len(out[s])} utts, {sum(len(x['audio']) for x in out[s]) / SR / 3600:.3f} h", flush=True)
    return out


def load_ookeval(cache_dir: str, limit=None):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(OOOK_REPO, OOOK_FILE, repo_type="dataset", revision=OOOK_REV, cache_dir=cache_dir)
    t = pq.read_table(p, columns=["id", "title", "text", "audio"]).to_pylist()
    out = []
    for r in t[: limit or None]:
        a = r["audio"]
        wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        out.append(dict(id=f"{r['id']}{r['title']}", text=(r["text"] or "").strip(), audio=_to16k(wav, sr)))
    print(f"[data] OOOK-Eval-Final: {len(out)} utts, {sum(len(x['audio']) for x in out) / SR / 3600:.3f} h", flush=True)
    return out
