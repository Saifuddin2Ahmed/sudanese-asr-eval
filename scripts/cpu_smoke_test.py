"""CPU smoke test of the whole training/decoding pipeline on a tiny subset, before spending GPU quota.

    pip install -e ".[experiments]"
    python scripts/cpu_smoke_test.py --work /tmp/sdn_smoke

Builds a mini SDN zip (6 clips per split, fetched from Zenodo by HTTP range reads),
then runs zero-shot decoding (whisper-tiny), a full fine-tune (whisper-tiny, 4 steps)
and a LoRA fine-tune (whisper-tiny, 4 steps), each end to end including the metrics
JSON. It also loads 4 OOOK-Eval-Final clips from the Hugging Face Hub (about 200 MB parquet).
The numbers are meaningless; only the plumbing is tested.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from data import SDN_FILES as FILES, open_remote_sdn_zip  # noqa: E402  (HTTP range reads with retries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--per_split", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    full_tsv = os.path.join(ROOT, "splits", "sdn_split_v1.tsv")
    lines = open(full_tsv, encoding="utf-8").read().splitlines()
    header = [x for x in lines if x.startswith("#")]
    body = [x for x in lines if not x.startswith("#")]
    cols, rows = body[0], body[1:]
    pick, count = [], {}
    for r in rows:
        s = r.split("\t")[1]
        if count.get(s, 0) < a.per_split:
            pick.append(r)
            count[s] = count.get(s, 0) + 1
    mini_tsv = os.path.join(a.work, "mini_split.tsv")
    open(mini_tsv, "w", encoding="utf-8").write("\n".join(header + [cols] + pick) + "\n")
    mini_zip = os.path.join(a.work, "mini_sdn.zip")
    if not os.path.exists(mini_zip):
        z = open_remote_sdn_zip()
        with zipfile.ZipFile(mini_zip, "w") as out:
            for r in pick:
                uid = r.split("\t")[0]
                for ext in (".wav", ".txt"):
                    out.writestr(FILES + uid + ext, z.read(FILES + uid + ext))
    print("mini zip:", mini_zip, count)

    def cfg(src, **over):
        c = json.load(open(os.path.join(ROOT, "experiments", "configs", src), encoding="utf-8"))
        c.update(split_tsv=mini_tsv, cache_dir=os.path.join(a.work, "cache"), out_dir=os.path.join(a.work, "out"),
                 bootstrap=100, **over)
        p = os.path.join(a.work, "smoke_" + src)
        json.dump(c, open(p, "w", encoding="utf-8"), indent=1)
        return p

    exp = os.path.join(ROOT, "experiments")
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "src"), exp] +
                                          ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    py = [sys.executable]
    zs = cfg("zeroshot.json", batch_size=4, models=[{"id": "openai/whisper-tiny"}])
    subprocess.run(py + ["run_zeroshot.py", "--config", zs, "--limit", "4", "--zip", mini_zip], cwd=exp, env=env,
                   check=True)
    ft = cfg("ft_whisper_small.json", name="smoke_full_tiny", base_model="openai/whisper-tiny", batch_size=2,
             eval_batch_size=4, warmup_steps=1, log_every=1, save_model=False)
    subprocess.run(py + ["run_finetune.py", "--config", ft, "--limit", "4", "--max_steps", "4", "--eval_every", "2",
                         "--zip", mini_zip], cwd=exp, env=env, check=True)
    lo = cfg("lora_whisper_large_v3_turbo.json", name="smoke_lora_tiny", base_model="openai/whisper-tiny",
             batch_size=2, grad_accum=2, eval_batch_size=4, warmup_steps=1, log_every=1, save_model=False)
    subprocess.run(py + ["run_finetune.py", "--config", lo, "--limit", "4", "--max_steps", "4", "--eval_every", "2",
                         "--zip", mini_zip], cwd=exp, env=env, check=True)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
