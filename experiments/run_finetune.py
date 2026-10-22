"""Fine-tune Whisper on the SDN speaker-disjoint TRAIN split.

    python experiments/run_finetune.py --config experiments/configs/ft_whisper_small.json [--limit 8 --max_steps 4]

Model selection uses the DEV split only (normalised WER, greedy decoding, every
`eval_every` steps; early stop after `patience` evals without improvement).
The TEST split and OOOK-Eval are decoded exactly ONCE, with the selected checkpoint,
after training ends. This fixes the audit finding where `load_best_model_at_end`
picked the checkpoint on test data.

mode = "full": all weights trained (fp32 master weights, fp16 autocast + GradScaler).
mode = "lora": PEFT LoRA adapters on attention + MLP projections of encoder and decoder;
               base frozen; gradient checkpointing to fit a 16 GB T4/P100.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

import asr
import data
from sudanese_asr_eval import metrics


def make_batches(n, bs, rng):
    idx = rng.permutation(n)
    return [idx[i:i + bs] for i in range(0, n - bs + 1, bs)]


def collate(proc, samples, device):
    feats = proc.feature_extractor([s["audio"] for s in samples], sampling_rate=16000,
                                   return_tensors="pt").input_features
    lab = [proc.tokenizer(s["text"]).input_ids for s in samples]
    start = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    lab = [(l[1:] if l and l[0] == start else l)[:448] for l in lab]  # model prepends decoder_start itself
    L = max(len(l) for l in lab)
    labels = torch.full((len(lab), L), -100, dtype=torch.long)
    for i, l in enumerate(lab):
        labels[i, :len(l)] = torch.tensor(l)
    return feats.to(device), labels.to(device)


def build_model(cfg, device):
    proc = WhisperProcessor.from_pretrained(cfg["base_model"], language="arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"], dtype=torch.float32)
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    if cfg.get("spec_augment"):
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = cfg.get("mask_time_prob", 0.05)
    if cfg["mode"] == "lora":
        from peft import LoraConfig, get_peft_model
        if cfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        lc = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                        target_modules=cfg["lora_targets"], bias="none")
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()
    elif cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.to(device), proc


@torch.no_grad()
def decode(model, proc, samples, bs, use_amp):
    model.eval()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.config.use_cache = True
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        hyps = asr.transcribe(model, proc, samples, batch_size=bs, log_every=0)
    base.config.use_cache = False
    model.train()
    return hyps


def trainable_state(model):
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
            if (not hasattr(model, "peft_config")) or "lora_" in k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--zip", default=None)
    args = ap.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    if args.eval_every:
        cfg["eval_every"] = args.eval_every
    out_dir = asr.ensure_dir(cfg["out_dir"])
    asr.set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and cfg.get("fp16", True)

    zip_path = args.zip or data.ensure_sdn_zip(cfg["cache_dir"])
    sdn = data.load_sdn(cfg["split_tsv"], zip_path, limit=args.limit)
    train, dev = sdn["train"], sdn["dev"]
    model, proc = build_model(cfg, device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    sched = get_linear_schedule_with_warmup(opt, cfg["warmup_steps"], cfg["max_steps"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(cfg["seed"])
    bs, accum = cfg["batch_size"], cfg.get("grad_accum", 1)

    log = open(os.path.join(out_dir, "train_log.jsonl"), "a", encoding="utf-8")
    best = dict(step=0, dev_wer_norm=math.inf)
    best_state, bad_evals, step, t0 = None, 0, 0, time.time()
    batches, run_loss = [], 0.0
    while step < cfg["max_steps"]:
        for _ in range(accum):
            if not batches:
                batches = make_batches(len(train), bs, rng)
            feats, labels = collate(proc, [train[i] for i in batches.pop()], device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(input_features=feats, labels=labels).loss / accum
            scaler.scale(loss).backward()
            run_loss += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg.get("max_grad_norm", 1.0))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        if step % cfg.get("log_every", 25) == 0:
            rec = dict(step=step, loss=round(run_loss / cfg.get("log_every", 25), 4), lr=sched.get_last_lr()[0],
                       elapsed=round(time.time() - t0))
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            run_loss = 0.0
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            hyps = decode(model, proc, dev, cfg["eval_batch_size"], use_amp)
            m = metrics.summarize([s["text"] for s in dev], hyps, B=200, seed=cfg["seed"])
            rec = dict(step=step, dev_wer_norm=m["wer_norm"], dev_cer_norm=m["cer_norm"], dev_wer_raw=m["wer_raw"],
                       elapsed=round(time.time() - t0))
            print("[dev]", rec, flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if m["wer_norm"] < best["dev_wer_norm"]:
                best = dict(step=step, dev_wer_norm=m["wer_norm"], dev=m)
                best_state = trainable_state(model)
                bad_evals = 0
            else:
                bad_evals += 1
                if bad_evals >= cfg.get("patience", 4):
                    print(f"[early stop] no dev improvement for {bad_evals} evals", flush=True)
                    break

    print(f"[select] best dev step {best['step']} (dev WER norm {best['dev_wer_norm']})", flush=True)
    model.load_state_dict(best_state, strict=False)
    if cfg.get("save_model", True):
        save_dir = os.path.join(out_dir, "best_model")
        model.save_pretrained(save_dir)
        proc.save_pretrained(save_dir)

    # ---- the ONLY time test data is decoded ----
    del sdn
    results = dict(env=asr.env_info(), config=cfg, selected_step=best["step"], dev=best.get("dev"),
                   train_utts=len(train), dev_utts=len(dev), train_seconds=round(time.time() - t0))
    test_sets = {"sdn_test": data.load_sdn(cfg["split_tsv"], zip_path, splits=("test",), limit=args.limit)["test"],
                 "ookeval": data.load_ookeval(cfg["cache_dir"], limit=args.limit)}
    for name, samples in test_sets.items():
        hyps = decode(model, proc, samples, cfg["eval_batch_size"], use_amp)
        asr.save_hyps(os.path.join(out_dir, f"hyps__{cfg['name']}__{name}.jsonl"), samples, hyps)
        results[name] = metrics.summarize([s["text"] for s in samples], hyps, B=cfg["bootstrap"], seed=cfg["seed"])
        r = results[name]
        print(f"[test] {name}: WER raw {r['wer_raw']} | basic {r['wer_basic']} | norm {r['wer_norm']} {r['wer_norm_ci95']}"
              f" CER norm {r['cer_norm']}", flush=True)
    json.dump(results, open(os.path.join(out_dir, f"{cfg['name']}_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[done]")


if __name__ == "__main__":
    main()
