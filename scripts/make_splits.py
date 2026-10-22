"""Build the speaker-disjoint train/dev/test split of the Sudanese dialect speech corpus.

Source: Mansour & Mukhtar, "Sudanese dialect speech dataset", Zenodo record 6869079
(DOI 10.5281/zenodo.6869079, v1.1, CC BY 4.0). The zip ships a per-utterance
`Metadata spreadsheet.csv` with a `speaker ID` column, so a true speaker-disjoint split
is possible.

Rules (all deterministic; durations are read from each WAV header with soundfile -
the corpus mixes 2,800 mono and 750 stereo 44.1 kHz files, so file size alone is wrong):
  * drop utterances with unknown speaker ("NaN"): they could be any speaker, so they
    could leak test speakers into training;
  * drop empty transcripts, and clips longer than 30 s (Whisper's window);
  * hm_17_0111.wav has no CSV row and no transcript -> not in the corpus;
  * search seeds 0..9999; for each seed shuffle the speakers and fill TEST, then DEV,
    greedily until each reaches its duration target; accept the FIRST seed for which
      - TEST holds 10-14 % of the kept duration, DEV 7-11 %,
      - TEST and DEV each contain >= 1 speaker labelled Female and both programmes
        (hm = Hajj Muzakir, wb = Dukkan Wad Elbaseer),
      - no single speaker is > 60 % of TEST or DEV duration,
      - the 3 largest speakers stay in TRAIN (otherwise a split is one voice).
The accepted seed and the stats are written into the TSV header.

Usage:
  python scripts/make_splits.py --zip /path/Sudanese_dialect_speech_dataset.zip --out splits/sdn_split_v1.tsv
  (rows are written with LF line endings, matching splits/sdn_split_v1.tsv)
  (--remote reads over HTTP range requests; Zenodo rate-limits this, so prefer --zip)
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import random
import sys
import urllib.request
import zipfile

ZENODO_URL = "https://zenodo.org/records/6869079/files/Sudanese_dialect_speech_dataset.zip?download=1"
ROOT = "SDN Dialect Corpus v1.0/"
FILES = ROOT + "Dataset files/"
MAX_SEC = 30.0


class HttpFile(io.RawIOBase):
    """Minimal seekable file over HTTP range requests (reads only what zipfile needs)."""

    def __init__(self, url: str):
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
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
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        data = urllib.request.urlopen(req).read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_zip(args) -> zipfile.ZipFile:
    if args.remote:
        return zipfile.ZipFile(io.BufferedReader(HttpFile(ZENODO_URL), buffer_size=1 << 16))
    return zipfile.ZipFile(args.zip)


def load_manifest(z: zipfile.ZipFile):
    infos = {i.filename: i for i in z.infolist()}
    meta = list(csv.DictReader(io.TextIOWrapper(z.open(ROOT + "Metadata spreadsheet.csv"), encoding="utf-8-sig")))
    rows, dropped = [], collections.Counter()
    for m in meta:
        uid = m["audio file name"].rsplit(".", 1)[0]
        wav, txt = FILES + uid + ".wav", FILES + uid + ".txt"
        if wav not in infos or txt not in infos:
            dropped["missing_file"] += 1
            continue
        text = z.read(txt).decode("utf-8-sig").strip()
        import soundfile as sf
        dur = sf.info(io.BytesIO(z.read(wav))).duration
        spk = m["speaker ID"].strip()
        if spk in ("", "NaN"):
            dropped["unknown_speaker"] += 1
            continue
        if not text:
            dropped["empty_text"] += 1
            continue
        if dur > MAX_SEC:
            dropped["longer_than_30s"] += 1
            continue
        rows.append(dict(id=uid, speaker=spk, gender=m["gender"].strip(), program=uid[:2], dur=round(dur, 3)))
    return rows, dropped, len(meta)


def try_seed(rows, seed):
    spk_dur = collections.Counter()
    spk_g = collections.defaultdict(set)
    spk_p = collections.defaultdict(set)
    for r in rows:
        spk_dur[r["speaker"]] += r["dur"]
        spk_g[r["speaker"]].add(r["gender"])
        spk_p[r["speaker"]].add(r["program"])
    total = sum(spk_dur.values())
    biggest = [s for s, _ in spk_dur.most_common(3)]
    cands = sorted(s for s in spk_dur if s not in biggest)
    random.Random(seed).shuffle(cands)
    test, dev = [], []
    t_d = d_d = 0.0
    for s in cands:
        if t_d < 0.12 * total:
            test.append(s); t_d += spk_dur[s]
        elif d_d < 0.09 * total:
            dev.append(s); d_d += spk_dur[s]
    ok = 0.10 <= t_d / total <= 0.14 and 0.07 <= d_d / total <= 0.11
    for part, dur in ((test, t_d), (dev, d_d)):
        ok &= any("Female" in spk_g[s] for s in part)
        ok &= {"hm", "wb"} <= set().union(*(spk_p[s] for s in part)) if part else False
        ok &= all(spk_dur[s] <= 0.6 * dur for s in part)
    return ok, set(test), set(dev), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip")
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    z = open_zip(args)
    rows, dropped, n_meta = load_manifest(z)
    for seed in range(10000):
        ok, test, dev, total = try_seed(rows, seed)
        if ok:
            break
    else:
        sys.exit("no seed satisfied the constraints")
    for r in rows:
        r["split"] = "test" if r["speaker"] in test else "dev" if r["speaker"] in dev else "train"
    stats = collections.defaultdict(lambda: [0, 0.0, set()])
    for r in rows:
        st = stats[r["split"]]; st[0] += 1; st[1] += r["dur"]; st[2].add(r["speaker"])
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(f"# source: Zenodo 10.5281/zenodo.6869079 v1.1 (CC BY 4.0); metadata rows {n_meta}\n")
        f.write(f"# dropped: {dict(dropped)}\n")
        f.write(f"# accepted seed: {seed}; kept {len(rows)} utts, {total / 3600:.3f} h\n")
        for k in ("train", "dev", "test"):
            n, d, s = stats[k]
            f.write(f"# {k}: {n} utts, {d / 3600:.3f} h ({100 * d / total:.1f} %), speakers {sorted(s, key=int)}\n")
        w = csv.DictWriter(f, fieldnames=["id", "split", "speaker", "gender", "program", "dur"], delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["id"]):
            w.writerow({k: r[k] for k in w.fieldnames})
    print(open(args.out, encoding="utf-8").read().split("id\tsplit")[0])


if __name__ == "__main__":
    main()
