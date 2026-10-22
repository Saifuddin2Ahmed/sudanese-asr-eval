"""Zero-shot evaluation of pretrained Whisper checkpoints.

    python experiments/run_zeroshot.py --config experiments/configs/zeroshot.json [--limit 4] [--models openai/whisper-tiny]

Writes <out_dir>/zeroshot_metrics.json and one hyps JSONL per (model, test set).
Test sets: SDN speaker-disjoint `test` split, and Mansour's OOOK-Eval-Final.
Dev is not scored here (zero-shot has nothing to select).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch

import asr
import data
from sudanese_asr_eval import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="tiny subset for smoke tests")
    ap.add_argument("--models", nargs="*", help="override model list (ids)")
    ap.add_argument("--zip", default=None, help="local SDN zip path, or 'remote' (else download)")
    args = ap.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    out_dir = asr.ensure_dir(cfg["out_dir"])
    asr.set_seed(cfg["seed"])
    if args.zip == "remote":  # HTTP range reads of only the needed clips (no 1.45 GB download)
        zip_path = data.open_remote_sdn_zip()
    else:
        zip_path = args.zip or data.ensure_sdn_zip(cfg["cache_dir"])
    sets = {"sdn_test": data.load_sdn(cfg["split_tsv"], zip_path, splits=("test",), limit=args.limit)["test"],
            "ookeval": data.load_ookeval(cfg["cache_dir"], limit=args.limit)}
    runs = cfg["models"] if not args.models else [dict(id=m) for m in args.models]
    path = os.path.join(out_dir, "zeroshot_metrics.json")
    results = json.load(open(path)) if os.path.exists(path) else {"env": asr.env_info(), "config": cfg, "runs": {}}
    for run in runs:
        name = run.get("name", run["id"])
        if name in results["runs"]:
            print(f"[skip] {name} already done")
            continue
        print(f"[model] {name}", flush=True)
        model, proc = asr.load(run["id"], run.get("gen_config_from"))
        res = {"model": run["id"], "language": run.get("language", "ar")}
        for set_name in run.get("sets", list(sets)):
            samples = sets[set_name]
            t0 = time.time()
            hyps = asr.transcribe(model, proc, samples, batch_size=run.get("batch_size", cfg["batch_size"]),
                                  language=run.get("language", "ar"))
            asr.save_hyps(os.path.join(out_dir, f"hyps__{name.replace('/', '__')}__{set_name}.jsonl"), samples, hyps)
            m = metrics.summarize([s["text"] for s in samples], hyps, B=cfg["bootstrap"], seed=cfg["seed"])
            m["seconds"] = round(time.time() - t0, 1)
            res[set_name] = m
            print(f"   {set_name}: WER raw {m['wer_raw']} | basic {m['wer_basic']} | norm {m['wer_norm']} "
                  f"{m['wer_norm_ci95']}  CER norm {m['cer_norm']}  ({m['seconds']}s)", flush=True)
        results["runs"][name] = res
        json.dump(results, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print("[done]", path)


if __name__ == "__main__":
    main()
