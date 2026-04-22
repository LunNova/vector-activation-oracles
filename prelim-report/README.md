# Prelim-report artefacts

Minimal reproduction + plotting for the ["trained steering vectors as AOs" post](https://lunnova.dev/articles/steering-vector-activation-oracle/).

## Layout

- `ckpts/` — trained Vector AO checkpoints + their companion training configs, their headline `*.results.json.zst` from `run_evals.py`, and the committed LoRA AO baseline run.
- `figures/` — `fig_eval_*.png` rendered by `make_figures.py`.
- `archived/` — hill-climb sweep variants (`pqa_*`, `lora_*`, `*_tp_think`), an older LoRA baseline snapshot (`lora_baseline_fa2_65pct.json.zst`), and a back-to-back rerun (`lora_rerun_B.json.zst`). See "LoRA-baseline y/n noise floor" below.

## Checkpoints (`ckpts/`)

- `full-mix-7.safetensors` / `.json` — AdamW full-mix Vector AO (294,912 params, post_attn + post_mlp).
- `scion-local02_steering_vectors_final.safetensors` / `.cfg` — Scion-optimizer Vector AO (headline figures in the post).

## Scripts

- `run_evals.py` — standalone HF-only runner. Takes a `.safetensors` of steering vectors plus its sibling config (`.json` or `.cfg`), runs Taboo + PersonaQA + PersonaQA y/n in the paper's headline configuration, writes `{stem}.results.json` next to the input. Use `--oracle-mode lora` + `--output` to run the LoRA AO baseline instead.
- `make_figures.py` — reads a Vector AO results JSON + LoRA baseline JSON, emits `{prefix}_taboo.png`, `{prefix}_personaqa.png`, and (when both carry `personaqa_primed`) `{prefix}_priming.png` into `figures/`.
- `make_worked_examples.py` — renders paired Vector / LoRA samples per task into `worked_examples.md` (the post's "Example Activation" section).
- `summarize_pqa_variants.py` — tabulates PQA open + y/n accuracy across collection-prompt variants (the post's priming table).
- `eval_data/` — copies of the paper's `taboo_direct_test.txt` and `personas.jsonl`, so `run_evals.py` has no dependency on `ref_submodules/`.

## Result JSONs

Stored as `*.results.json.zst` (zstd level 19, ~107x compression — each full-sized JSON has per-item detail arrays full of repeated chat-template text). `io_utils.load_json()` transparently accepts either the plain `.json` (if `run_evals.py` has just written one) or the committed `.zst`. To inspect manually: `zstd -dc foo.results.json.zst | jq .`. A fresh `run_evals.py` output is gitignored — re-compress with `zstd -19 --rm` before committing.

Default (paper-faithful) collection in `ckpts/`:

- `full-mix-7.results.json.zst` — AdamW Vector AO baseline.
- `scion-local02_steering_vectors_final.results.json.zst` — Scion Vector AO baseline (both files also embed a `personaqa_primed` block, used for the `_priming.png` figure).
- `lora_baseline_fa2.json.zst` — LoRA AO baseline (matches the paper's config: flash_attention_2, per-item SET injection, set-order distractors).

Collection-prompt variants for the priming table live in `archived/` (one result file per `(oracle, prompt)` cell):

- `pqa_t_minimal.results.json.zst`, `lora_t_minimal.results.json.zst` — `<think>` = `"{name}: country, food, drink, music genre, sport, boardgame."`
- `pqa_a_compact.results.json.zst`, `lora_a_compact.results.json.zst` — answer = `"What are {name}'s favorite country, food, drink, music genre, sport, and boardgame?"`
- `pqa_a_decl.results.json.zst`, `lora_a_decl.results.json.zst` — answer = `"{name}'s favorite country, food, drink, music genre, sport, and boardgame."`
- `scion_tp_think.results.json.zst`, `fullmix7_tp_think.results.json.zst`, `lora_thirdperson_think.results.json.zst` — `<think>` = `"The user is {name}. They have specific preferences to recall."`

## LoRA-baseline y/n noise floor

PersonaQA y/n scoring under `run_evals.py` is deterministic on the yes-side but picks "no"-side distractors via `random.choice(list(remaining))` where `remaining` is a `set` — iteration order depends on the per-process `PYTHONHASHSEED`. Across four fresh runs we saw the LoRA AO baseline land at 65.1%, 66.8%, 67.3%, 68.0% on y/n (Taboo and PQA-open stayed bit-exact). The committed `ckpts/lora_baseline_fa2.json.zst` is the highest of those runs (68.0%); `archived/lora_baseline_fa2_65pct.json.zst` is a 65.1% run and `archived/lora_rerun_B.json.zst` is a 67.3% run, kept so the noise spread is documented. The same bug is present in the paper's `personaqa_yes_no_eval.py:247-252`, which means the paper's numbers have the same noise floor.

## Usage

```
# Produce results (needs GPU; loads Qwen3-8B; pulls task LoRAs from HF Hub)
nix develop ".#cuda" -c python3 prelim-report/run_evals.py prelim-report/ckpts/scion-local02_steering_vectors_final.safetensors

# Produce per-checkpoint figures (CPU; default devshell includes matplotlib + xkcd-font)
nix develop -c python3 prelim-report/make_figures.py prelim-report/ckpts/scion-local02_steering_vectors_final.results.json.zst \
    --name fig_eval_scion-local02_final

# Rebuild the post's "Example Activation" markdown
nix develop -c python3 prelim-report/make_worked_examples.py \
    --vector prelim-report/ckpts/scion-local02_steering_vectors_final.results.json.zst \
    --baseline prelim-report/ckpts/lora_baseline_fa2.json.zst

# Rebuild the priming table
nix develop -c python3 prelim-report/summarize_pqa_variants.py \
    --baseline prelim-report/ckpts/scion-local02_steering_vectors_final.results.json.zst \
    prelim-report/archived/pqa_t_minimal.results.json.zst prelim-report/archived/pqa_a_compact.results.json.zst \
    prelim-report/archived/pqa_a_decl.results.json.zst prelim-report/archived/scion_tp_think.results.json.zst
```
