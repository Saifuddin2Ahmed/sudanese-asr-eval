# sudanese-asr-eval

Code, split and scores for the paper **"What Is in a WER? Reproducible Evaluation of Whisper for Sudanese Arabic"** (Saifuddin Ahmed, 2026).

- **Paper:** arXiv link to be added (TBD).
- **Archive:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24354752.svg)](https://doi.org/10.5281/zenodo.24354752)
- **Repository:** https://github.com/Saifuddin2Ahmed/sudanese-asr-eval

The repository contains:

| Path | What it is |
|---|---|
| `src/sudanese_asr_eval/` | Installable Python module: the three text views (`raw`, `basic`, `norm`), corpus WER/CER with bootstrap CIs, paired bootstrap tests and the loop-collapse diagnostic. Includes the `sudanese-asr-eval` command-line tool. |
| `splits/sdn_split_v1.tsv` | The speaker-disjoint train/dev/test split of the Sudanese dialect speech corpus. It holds clip IDs, speaker IDs, gender, programme and duration only, with no audio and no transcripts. |
| `experiments/` | Data loading, Whisper decoding, the zero-shot runner, the full fine-tuning and LoRA runner, and the exact configs used. |
| `notebooks/kaggle/` | The three Kaggle GPU notebooks, generated from the code above. The Kaggle username is a placeholder and they contain no credentials. |
| `scripts/` | Utility scripts:<br>• `reproduce_paper.py`: recomputes every number in the paper;<br>• `export_per_clip.py`: hypotheses to per-clip counts;<br>• `check_mirror_alignment.py`: the mirror check;<br>• `make_splits.py`: rebuilds the split;<br>• `build_notebooks.py`: generates the notebooks;<br>• `cpu_smoke_test.py`: an end-to-end check on CPU. |
| `results/` | Released scores:<br>• `metrics/`: the metrics JSONs written on Kaggle;<br>• `per_clip/`: text-free per-clip error counts for every system and both test sets;<br>• `hyps_sdn_test/`: SDN-test hypotheses;<br>• `train_logs/`: dev curves;<br>• `paper_numbers.json`: the output of `reproduce_paper.py`. |
| `tests/` | Unit tests (`python -m unittest discover -s tests`). |

## Install

```bash
pip install .                    # the normaliser, scoring and CLI (needs numpy and jiwer)
pip install ".[experiments]"     # also the pinned training/decoding stack
```

You need Python 3.9 or newer.
- The scoring code gives identical numbers with jiwer 3.1.0 (used for the paper) and jiwer 4.0.
- The training and decoding stack is pinned to `transformers==4.56.2`, `peft==0.17.1` and `jiwer==3.1.0`.

## The three text views

```python
from sudanese_asr_eval import raw, basic, full

s = "ألف و تسعمية تمنية، يا زول!"
raw(s)    # 'ألف و تسعمية تمنية، يا زول!'   Unicode NFC + whitespace only
basic(s)  # 'ألف و تسعمية تمنية يا زول'     Whisper BasicTextNormalizer (no letter unification)
full(s)   # 'الف وتسعميه تمنيه يا زول'      full Arabic normaliser (the paper's "norm")
```

The `full` view applies these steps:
- NFKC;
- removes diacritics, tatweel and Quranic marks;
- maps أ إ آ ٱ → ا, ى → ي, ؤ → و, ئ → ي and ة → ه;
- converts Eastern-Arabic and Persian digits to ASCII;
- removes punctuation and symbols;
- lower-cases Latin letters;
- attaches a stand-alone conjunction و to the next word.

The same view is always applied to both the reference and the hypothesis. Each step is documented in `src/sudanese_asr_eval/textnorm.py`.

### Command line

```bash
echo "ألف و تسعمية" | sudanese-asr-eval normalize                 # -> الف وتسعميه
sudanese-asr-eval normalize --view all transcripts.txt           # TSV: input, raw, basic, norm
sudanese-asr-eval score results/hyps_sdn_test/lora_turbo.jsonl   # WER/CER, 3 views, 95 % CIs
sudanese-asr-eval compare results/hyps_sdn_test/lora_turbo.jsonl results/hyps_sdn_test/zs_turbo.jsonl --view raw
```

- **Input format:** hypothesis files are JSON Lines with `id`, `ref` and `hyp`.
- **Bootstrap:** 1,000 utterance resamples with seed 42, as in the paper.
- **`compare`:** pairs clips by `id` and reports WER(A) − WER(B) with a paired-bootstrap 95 % CI.

## Reproducing the numbers

### Step 1: every number in the paper, from the released counts (CPU, about a minute)

```bash
python -m unittest discover -s tests
python scripts/reproduce_paper.py --check
```

`reproduce_paper.py` reads `results/per_clip/` and prints each section of the paper. It also writes them to `results/paper_numbers.json`.

| Output section | Paper |
|---|---|
| `main` | Table 1: WER and CER, raw / basic / norm, with 95 % CIs, for 9 systems on both test sets |
| `gain` | Normalisation effect (raw − norm WER), §4 and abstract |
| `paired` | Paired bootstrap comparisons, §4 |
| `ablation` | Appendix B: cumulative normalisation steps |
| `halluc` | Hallucinated clips and their share of word errors, §4 "Error analysis" |
| `loops` | Exploratory loop-collapse scores, §4 (post hoc, not a result) |
| `speakers` | Per-speaker WER on the SDN test set, Limitations |

With `--check`, the script compares every main-table value (point estimates and CIs, all three views, WER and CER) with the metrics JSONs written on Kaggle. It exits with an error on any difference.

The SDN-test hypotheses are included, so these numbers can also be re-scored from text:

```bash
sudanese-asr-eval score results/hyps_sdn_test/lora_turbo.jsonl
```

### Step 2: re-decode and re-train (Kaggle T4, about 10 GPU-hours)

1. **Install and authenticate the Kaggle CLI.** Run `pip install kaggle`, then put your own API token in `~/.kaggle/kaggle.json`. Never commit the token.
2. **Build the notebooks for your account:**
   `python scripts/build_notebooks.py --user <your-kaggle-username>`
   - Kernels are private by default.
   - A Hugging Face token is optional. If you store one as a Kaggle secret named `HF_TOKEN`, the notebooks use it.
3. **Push each kernel.** Pushing starts the run, and Kaggle allows two GPU sessions at once.
   ```bash
   kaggle kernels push -p notebooks/kaggle/sudanese-asr-zeroshot          # 7 zero-shot runs, ~1 h
   kaggle kernels push -p notebooks/kaggle/sudanese-asr-ft-whisper-small  # full FT, ~1.5 h
   kaggle kernels push -p notebooks/kaggle/sudanese-asr-lora-turbo        # LoRA, ~7 h
   ```
   Choose the T4 accelerator. The paper's runs used a T4 in fp16.
4. **Download the outputs:**
   `kaggle kernels output <user>/<slug> -p outputs/<slug> --file-pattern ".*\.(json|jsonl|txt|tsv)$"`
5. **Rebuild the counts and the tables:**
   ```bash
   python scripts/export_per_clip.py outputs --out results/per_clip
   cp outputs/*/*_metrics.json results/metrics/
   python scripts/reproduce_paper.py --check
   ```

What the notebooks do:
- Each one downloads the corpus from Zenodo and checks its MD5.
- Each one loads OOOK-Eval-Final from the Hugging Face Hub at revision `e256ab538b6efb2e0e9e452b962d44b198662619`.
- Each one runs the normaliser unit tests (including parity with Transformers' `BasicTextNormalizer`) before the experiment.
- Each one saves a `pip freeze` with its outputs.

What to expect from a re-run:
- GPU decoding is not bit-exact across drivers and library versions. Expect small differences: in our runs, fp32 on CPU and fp16 on GPU differed by at most 0.5 WER.
- Before spending GPU time, `python scripts/cpu_smoke_test.py --work <dir>` runs the whole pipeline on CPU with whisper-tiny on 18 clips.

**Protocol:**
- greedy decoding, Arabic language token, `max_new_tokens=225`, fp16;
- split seed 354, training seed 42, bootstrap seed 42 (1,000 resamples);
- checkpoints selected on dev WER (norm view) only, with each test set decoded once.

All hyper-parameters are in `experiments/configs/`.

### The split

To rebuild the split from the Zenodo v1.1 archive:

```bash
python scripts/make_splits.py --zip Sudanese_dialect_speech_dataset.zip --out splits/sdn_split_v1.tsv
```

**What was dropped:**
- 182 clips without a speaker ID, because they could leak test voices into training;
- one clip longer than 30 s.

| Split | Clips | Hours | Speakers |
|---|---|---|---|
| train | 2,695 | 3.01 | 14 |
| dev | 302 | 0.39 | 15 |
| test | 369 | 0.47 | 5: IDs 6, 10, 17, 20 and 26 |

The test set has only five speakers, so treat small differences on it with care (see the paper's Limitations).

### The mirror check

```bash
python scripts/check_mirror_alignment.py mirror_check/            # or add --zip <local Zenodo v1.1 zip>
```

This compares every row of the Hugging Face mirror `arbml/sudanese_dialect_speech` with the original Zenodo transcripts.
- **Our result (October 2026):** only 85 of 3,547 rows carry the transcript of their own audio file. 1,554 carry the next clip's transcript and 1,907 the one after that.
- **Recommendation:** use the original Zenodo release.

## Data sources and licences

**Sudanese dialect speech corpus.** Ayman Mansour (supervised by Wafaa F. Mukhtar), Sudan University of Science and Technology, "Sudanese dialect speech dataset", Zenodo, licensed **CC BY 4.0**.
- **Releases:**
  - Latest release: [Zenodo record 7426918](https://zenodo.org/records/7426918) (v1.2).
  - All versions: concept DOI [10.5281/zenodo.6721682](https://doi.org/10.5281/zenodo.6721682).
- **Version used here:** every number uses **v1.1**, [record 6869079](https://zenodo.org/records/6869079) (doi:10.5281/zenodo.6869079), whose archive MD5 (`61effea8ff14daa690bb0e05f9930719`) the code checks.
  - v1.2 is a different archive and has not been evaluated here.
  - The split refers to clip IDs of v1.1.
- **Attribution for our derived files:** "Sudanese dialect speech dataset" by Ayman Mansour (supervised by Wafaa F. Mukhtar), Zenodo, doi:10.5281/zenodo.6869079, licensed CC BY 4.0. Changes: clips without a speaker ID and one clip longer than 30 s were removed, a speaker-disjoint split was added, and machine transcripts and error counts were added.

**OOOK-Eval-Final.** Hugging Face `AymanMansour/OOOK-Eval-Final`, introduced by Mansour (2026, arXiv:2601.06802) and drawn from the Sudanese part of ADI17 (Shon et al., 2020).
- It is used **for evaluation only** and is **not redistributed** here.
- This repository contains no OOOK-Eval audio, transcripts or hypotheses. It contains only clip IDs and numeric per-clip error counts, in `results/per_clip/*__ookeval.tsv`.
- To score on it, download it from its source under its own terms.

**Models.**
- Whisper checkpoints are by OpenAI.
- `AymanMansour/Whisper-Sudanese-Dialect-small` is the public 2022 checkpoint, scored here as an external reference.

## Licences

| What | Licence |
|---|---|
| Code: `src/`, `experiments/`, `scripts/`, `tests/`, `notebooks/` | Apache License 2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)) |
| Split and released scores: `splits/`, `results/` | CC BY 4.0 ([`LICENSE-DATA`](LICENSE-DATA)) |

The CC BY 4.0 licence does not extend to OOOK-Eval-Final.

## Citation

If you use the split, normaliser or scores, please cite the paper and the corpus. Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

```bibtex
@misc{ahmed2026wer,
  author = {Saifuddin Ahmed},
  title  = {What Is in a {WER}? Reproducible Evaluation of {W}hisper for {S}udanese {A}rabic},
  year   = {2026},
  note   = {Preprint; arXiv identifier to be added},
  url    = {https://github.com/Saifuddin2Ahmed/sudanese-asr-eval}
}

@dataset{mansour_2022_sdn,
  author    = {Ayman Mansour},
  title     = {Sudanese dialect speech dataset},
  year      = {2022},
  publisher = {Zenodo},
  version   = {1.1},
  doi       = {10.5281/zenodo.6869079},
  note      = {Supervised by Wafaa F. Mukhtar; CC BY 4.0}
}

@article{mansour2022e2e,
  author  = {Ayman Mansour and Wafaa F. Mukhtar},
  title   = {End-to-End Automatic Speech Recognition model for the {S}udanese Dialect},
  journal = {arXiv preprint arXiv:2212.10826},
  year    = {2022}
}

@article{mansour2026doing,
  author  = {Ayman Mansour},
  title   = {Doing More with Less: Data Augmentation for {S}udanese Dialect Automatic Speech Recognition},
  journal = {arXiv preprint arXiv:2601.06802},
  year    = {2026}
}
```

## Acknowledgements

This work rests on Ayman Mansour's Sudanese dialect speech corpus, his OOOK-Eval set and his public checkpoints.
