"""Check whether the Hugging Face mirror `arbml/sudanese_dialect_speech` pairs each audio
file with its own transcript, by comparing it with the original Zenodo release (v1.1).

    python scripts/check_mirror_alignment.py OUT_DIR [--zip Sudanese_dialect_speech_dataset.zip]

Standard library only. Steps:
  1. read (row index, file name from the `path` column, text) for every mirror row through the
     Hugging Face datasets-server API (no audio is downloaded);
  2. read every transcript (.txt) of the Zenodo zip, either from a local copy (--zip) or from
     the remote zip by HTTP range requests (about 3,500 small reads, no 1.45 GB download);
  3. for each mirror row, test whether its text equals the Zenodo transcript of the same file,
     or of a file at offset +1, +2, +3, -1, -2, +4, +5 in Zenodo file order.

Writes OUT_DIR/summary.json and OUT_DIR/examples.json. Steps 1 and 2 are cached in OUT_DIR
(arbml_rows.json, zenodo_txt.json), so a re-run is offline. Result on 2026-10-22: 85 of 3,547
rows match the same file; 1,554 match the next clip and 1,907 the clip after that.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import struct
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

ZENODO_URL = "https://zenodo.org/records/6869079/files/Sudanese_dialect_speech_dataset.zip?download=1"
ZENODO_MD5 = "61effea8ff14daa690bb0e05f9930719"
ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=arbml/sudanese_dialect_speech"
            "&config=default&split=train&offset={o}&length=100")
N_MIRROR_ROWS = 3547
OFFSETS = (1, 2, 3, -1, -2, 4, 5)


def get(url, headers=None, tries=6):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120).read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 + 3 * i)


def mirror_rows(cache):
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    rows = []
    for o in range(0, N_MIRROR_ROWS, 100):
        d = json.loads(get(ROWS_URL.format(o=o)))
        for r in d["rows"]:
            rows.append({"row_idx": r["row_idx"], "file": os.path.basename(r["row"]["path"]), "text": r["row"]["text"]})
        time.sleep(0.5)
    json.dump(rows, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    return rows


class HttpFile(io.RawIOBase):
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
        d = get(self.url, {"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def _is_txt(info):
    return info.filename.endswith(".txt") and "/Dataset files/" in info.filename


def _wav_name(info):
    return os.path.basename(info.filename)[:-4] + ".wav"


def zenodo_texts(cache, local_zip=None):
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    if local_zip:
        import hashlib
        h = hashlib.md5()
        with open(local_zip, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != ZENODO_MD5:
            raise SystemExit(f"zip md5 {h.hexdigest()} != {ZENODO_MD5} (expected Zenodo v1.1)")
        z = zipfile.ZipFile(local_zip)
        texts = {_wav_name(i): z.read(i).decode("utf-8-sig").strip() for i in z.infolist() if _is_txt(i)}
    else:
        hf = HttpFile(ZENODO_URL)
        z = zipfile.ZipFile(io.BufferedReader(hf, buffer_size=1 << 16))
        infos = [i for i in z.infolist() if _is_txt(i)]

        def read_member(info):
            # local header = 30 bytes + name + extra field; fetch header and data in one range request
            start = info.header_offset
            blob = get(hf.url, {"Range": f"bytes={start}-{start + 30 + len(info.filename.encode()) + 200 + info.compress_size}"})
            n, m = struct.unpack("<HH", blob[26:30])
            data = blob[30 + n + m: 30 + n + m + info.compress_size]
            if info.compress_type == zipfile.ZIP_DEFLATED:
                data = zlib.decompress(data, -15)
            return _wav_name(info), data.decode("utf-8-sig").strip()

        with ThreadPoolExecutor(12) as ex:
            texts = dict(ex.map(read_member, infos))
    json.dump(texts, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    return texts


def compare(rows, zen):
    def norm(s):
        return " ".join(s.split())

    order = sorted(zen)  # Zenodo file order: hm_01_0001 ... wb_03_xxxx
    pos = {f: i for i, f in enumerate(order)}
    exact, shift, examples = 0, collections.Counter(), []
    for r in rows:
        f, t = r["file"], norm(r["text"])
        if f not in zen:
            shift["file_missing_in_zenodo"] += 1
            continue
        if norm(zen[f]) == t:
            exact += 1
            continue
        found = next((d for d in OFFSETS if 0 <= pos[f] + d < len(order) and norm(zen[order[pos[f] + d]]) == t), None)
        shift[found] += 1
        if len(examples) < 400:
            examples.append({"row_idx": r["row_idx"], "file": f, "mirror_text": t, "zenodo_text_same_file": norm(zen[f]),
                             "matches_file_at_offset": found,
                             "matched_file": order[pos[f] + found] if found else None})
    summary = {"mirror_rows": len(rows), "zenodo_transcripts": len(zen), "exact_same_file": exact,
               "mismatch_by_offset": {str(k): v for k, v in shift.items()}}
    return summary, examples


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--zip", help="local copy of the Zenodo v1.1 zip (md5-checked)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows = mirror_rows(os.path.join(a.out_dir, "arbml_rows.json"))
    zen = zenodo_texts(os.path.join(a.out_dir, "zenodo_txt.json"), a.zip)
    summary, examples = compare(rows, zen)
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(examples, open(os.path.join(a.out_dir, "examples.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
