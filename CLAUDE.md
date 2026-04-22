# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Dev shells (nix)

- `nix develop` — CPU-only shell (matplotlib, numpy, transformers, etc.). Enough for `prelim-report/make_*.py` and most data wrangling.
- `nix develop ".#cuda"` — CUDA shell with torch+CUDA, accelerate, peft, bitsandbytes, and a rebuilt `flash-attn` (compiled for sm_80; forward-compat to Ada sm_89). Needed by `prelim-report/run_evals.py` (loads Qwen3-8B via HF with `attn_implementation="flash_attention_2"`). `.envrc` auto-selects this via direnv.
- `nix develop ".#all"` — ROCm shell built for all supported GPU archs. **Prefer this for ROCm work** when it's a cache hit: larger closure but avoids ~1000 core-hour ROCm rebuilds. Training (`python -m src.run_experiment`) runs from here. Fall back to arch-specific shells (`.#gfx90a`, `.#gfx942`, `.#gfx950`) only when `.#all` isn't cached and you need a smaller local build. Exports `ROCM_PATH`, `GPU_ARCH*`.

Always quote flake refs: `nix develop ".#cuda"` (unquoted `.#` breaks permission matching).

## Common commands

```bash
# One-time asset download (writes to sibling dir ../vector-activation-oracles-data/, symlinks models/<name>/)
python scripts/download_assets.py

# Train a steering vector AO (from a ROCm shell)
python -m src.run_experiment --config configs/full_mix.json
# CLI overrides most ExperimentConfig fields: --lr, --optimizer {adamw,natural_grad,spectral_scion,ademamix}, --injection-points, --layers, etc.
# See src/run_experiment.py for the full list.

# Standalone eval of a trained checkpoint
python -m src.eval_harness --config configs/first_experiment.json \
    --vectors path/to/steering_vectors.safetensors \
    --classification  # or --taboo, --personaqa

# Prelim-report pipeline (self-contained, HF-only — does not import src/)
nix develop ".#cuda" -c python3 prelim-report/run_evals.py \
    prelim-report/ckpts/scion-local02_steering_vectors_final.safetensors
nix develop -c python3 prelim-report/make_figures.py \
    prelim-report/ckpts/scion-local02_steering_vectors_final.results.json.zst \
    --name fig_eval_scion-local02_final

# Lint (ruff is in every devShell)
ruff check src/ prelim-report/ scripts/
```

There is no test suite and no single-test runner.

## Data layout

Assets live in a **sibling directory** `../vector-activation-oracles-data/` (configured via `ExperimentConfig.data_dir`), not under this repo — this avoids copying multi-GB blobs into the nix store. Expected subtree:

```
../vector-activation-oracles-data/
    models/Qwen3-8B/           # symlink into hf_cache/ (base model for training + eval)
    models/Qwen3-8B_ao/        # symlink into hf_cache/ (adamkarvonen LatentQA checkpoint)
    hf_cache/                   # HF hub cache, incl. 20 `Qwen3-8B-taboo-{word}` LoRAs
    datasets/                   # HF datasets cache + md_gender_funpedia/
    activation_cache/<model>_<dtype>/  # per-config mmap-backed cached activations
    outputs/<run-name>/         # training checkpoints (steering_vectors_*.safetensors)
```

Classification CSVs (`geometry_of_truth`, `relations`, `tense`, `singular_plural`, `ner`, `ag_news`, `engels`) come from the `ref_submodules/activation_oracles` git submodule, not HF. Don't forget `git submodule update --init`.

## Architecture

### Custom Qwen3 implementation (`src/model.py`)

`OracleTransformer` is a **from-scratch Qwen3 reimplementation**, not an HF wrapper. Chosen so the training loop can cheaply inject steering vectors at `post_attn` and/or `post_mlp` residual points on every block and inject collected activations at a fixed `injection_layer`. Key details:

- Base weights loaded via `src/weights.py::load_weights` which maps HF keys (`model.embed_tokens`, `model.layers.*.self_attn.*`, etc.) to this model's naming (`embed`, `blocks.*.attn.*`, `norm_1`, `norm_2`).
- Base params are materialized from a meta-device init; steering vectors are real tensors and only those get gradients (`freeze_base()`).
- `share_base_weights(oracle, hf_model)` aliases base parameters to an HF model's params — lets `src/eval_harness.py` run eval against the HF LatentQA checkpoint without doubling 8B of bf16 memory.
- Attention has **two paths**: SDPA for bf16/f16 and manual softmax for f32. The f32 path works around a ROCm 7.2 / PyTorch 2.10 SDPA bug on MI210 (`scripts/investigate_sdpa_bug.py`, referenced in code). Don't "simplify" by removing the manual path.
- `generate()` implements greedy decode with a KV cache; steering is applied at prefill **and** decode, activation injection only at prefill. `collect_activations_multi()` extracts hidden states at multiple layers in one forward pass for the activation-cache precompute.

### Training pipeline (`src/train.py`, `src/data.py`, `src/optim.py`)

- Three **data sources**, mixed per config: classification (10 datasets, generated as multi-turn multi-QA chat prompts with `PLACEHOLDER = " ?"` tokens as injection slots), context prediction (predict tokens before/after an activation window over FineWeb/LMSYS), and SPQA (LatentQA-style system-prompt QA). `ExperimentConfig.classification_variants` and `context_prediction_variants` control expansion.
- **Pre-computed activations**: `precompute_activations()` runs the frozen base model once over all training contexts, writes per-example tensors to `activation_cache/<model>_<dtype>/`, then `load_activation_cache()` re-hydrates via mmap-backed safetensors. Skip-on-exists so re-runs of the same config are fast.
- Steering vectors are per `(injection_point, layer_idx)` parameters of shape `(n_embd,)`. Injection is **norm-matched**: `h' = h + ||h|| * normalize(v) * coeff` (not raw addition) — `OracleTransformer._inject_activations`.
- Custom optimizers in `src/optim.py`: `NaturalGradient` (per-vector empirical Fisher, O(D²) state), `SpectralScion` (Newton-Schulz orthogonalization over stacked (N_layers, D) matrix), `AdEMAMix` (fast+slow EMA mix). `adamw` is the default. The Scion optimizer produced the headline `scion-local02_steering_vectors_final` checkpoint.
- LR schedules: `wsd` (warmup-stable-decay, default) or `cosine`.

### Eval (`src/eval.py`, `src/eval_harness.py`)

- `src/eval.py::evaluate` runs during training against the held-out split, partitioned id/ood by `trained_dataset_names`.
- `src/eval_harness.py` is a CLI for post-hoc eval on a checkpoint. Supports classification, Taboo (three activation-input modes: `assistant_sot`, `segment`, `full_seq`), and PersonaQA. Can load vectors from safetensors with injection_points recovered from the file's metadata, or inferred from tensor keys.
- `run_taboo_probe` (training-time) loads `adamkarvonen/Qwen3-8B-taboo-{word}_50_mix` LoRAs and checks whether the oracle can surface the hidden word.

### `prelim-report/` — blog-post artifacts (self-contained)

Separate from `src/`. `run_evals.py`, `make_figures.py`, `make_worked_examples.py`, `summarize_pqa_variants.py` use **HuggingFace models only** (no `src.model` import) so the post's minimal repro doesn't depend on the custom Qwen3 stack. `eval_data/` has copies of `taboo_direct_test.txt` and `personas.jsonl` so it doesn't need `ref_submodules/` either.

- Subdirs: `ckpts/` (trained Vector AO safetensors + sidecar training configs + their headline `*.results.json.zst` + the committed `lora_baseline_fa2.json.zst`), `figures/` (`fig_eval_*.png` from `make_figures.py`), `archived/` (hill-climb sweep variants, older LoRA baseline snapshots, A/B reruns).
- Two checkpoints checked in under `ckpts/`: `full-mix-7.safetensors` (AdamW baseline) and `scion-local02_steering_vectors_final.safetensors` (Scion, headline).
- Results are committed as `*.results.json.zst` (zstd -19). `io_utils.load_json()` accepts either `.json` (fresh) or `.zst` (committed). After running `run_evals.py`, recompress with `zstd -19 --rm` before committing; the plain `.json` is gitignored.
- `run_evals.py`'s y/n distractor sampling is `PYTHONHASHSEED`-dependent (distractors pulled from a `set`, pattern inherited from the paper's `personaqa_yes_no_eval.py`). Fresh runs drift the LoRA-baseline y/n number by ~3pp; the committed `ckpts/lora_baseline_fa2.json.zst` is the highest of four observed runs (68.0%).
