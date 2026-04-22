# Prelim-report artefacts

Minimal reproduction + plotting for the ["trained steering vectors as AOs" post](https://lunnova.dev/articles/steering-vector-activation-oracle/).

## Checkpoints

- `full-mix-7.safetensors` / `.json` — AdamW full-mix Vector AO (294,912 params, post_attn + post_mlp).
- `scion-local02_steering_vectors_final.safetensors` / `.cfg` — Scion-optimizer Vector AO (headline figures in the post).

## Scripts

- `run_evals.py` — standalone HF-only runner. Takes a `.safetensors` of steering vectors plus its sibling config (`.json` or `.cfg`), runs Taboo + PersonaQA + PersonaQA y/n in the paper's headline configuration, writes `{stem}.results.json`. Use `--oracle-mode lora` to run the LoRA AO baseline instead.
- `make_figures.py` — reads a Vector AO results JSON + LoRA baseline JSON, emits `{prefix}_taboo.png`, `{prefix}_personaqa.png`, and (when both carry `personaqa_primed`) `{prefix}_priming.png`.
- `make_worked_examples.py` — renders paired Vector / LoRA samples per task into `worked_examples.md` (the post's "Example Activation" section).
- `summarize_pqa_variants.py` — tabulates PQA open + y/n accuracy across collection-prompt variants (the post's priming table).
- `eval_data/` — copies of the paper's `taboo_direct_test.txt` and `personas.jsonl`, so `run_evals.py` has no dependency on `ref_submodules/`.

## Result JSONs

Stored as `*.results.json.zst` (zstd level 19, ~107x compression — each full-sized JSON has per-item detail arrays full of repeated chat-template text). `io_utils.load_json()` transparently accepts either the plain `.json` (if `run_evals.py` has just written one) or the committed `.zst`. To inspect manually: `zstd -dc foo.results.json.zst | jq .`. A fresh `run_evals.py` output is gitignored — re-compress with `zstd -19 --rm` before committing.

Default (paper-faithful) collection:

- `full-mix-7.results.json` — AdamW Vector AO baseline.
- `scion-local02_steering_vectors_final.results.json` — Scion Vector AO baseline (both files also embed a `personaqa_primed` block, used for the `_priming.png` figure).
- `lora_baseline_fa2.json` — LoRA AO baseline (matches the paper's config: flash_attention_2, per-item SET injection, set-order distractors).

Collection-prompt variants for the priming table (one result file per `(oracle, prompt)` cell):

- `pqa_t_minimal.results.json`, `lora_t_minimal.results.json` — `<think>` = `"{name}: country, food, drink, music genre, sport, boardgame."`
- `pqa_a_compact.results.json`, `lora_a_compact.results.json` — answer = `"What are {name}'s favorite country, food, drink, music genre, sport, and boardgame?"`
- `pqa_a_decl.results.json`, `lora_a_decl.results.json` — answer = `"{name}'s favorite country, food, drink, music genre, sport, and boardgame."`
- `scion_tp_think.results.json`, `fullmix7_tp_think.results.json`, `lora_thirdperson_think.results.json` — `<think>` = `"The user is {name}. They have specific preferences to recall."`

## Usage

```
# Produce results (needs GPU; loads Qwen3-8B; pulls task LoRAs from HF Hub)
nix develop ".#cuda" -c python3 prelim-report/run_evals.py prelim-report/scion-local02_steering_vectors_final.safetensors

# Produce per-checkpoint figures (CPU; default devshell includes matplotlib + xkcd-font)
nix develop -c python3 prelim-report/make_figures.py prelim-report/scion-local02_steering_vectors_final.results.json \
    --name fig_eval_scion-local02_final

# Rebuild the post's "Example Activation" markdown
nix develop -c python3 prelim-report/make_worked_examples.py \
    --vector prelim-report/scion-local02_steering_vectors_final.results.json \
    --baseline prelim-report/lora_baseline_fa2.json

# Rebuild the priming table
nix develop -c python3 prelim-report/summarize_pqa_variants.py \
    --baseline prelim-report/scion-local02_steering_vectors_final.results.json \
    prelim-report/pqa_t_minimal.results.json prelim-report/pqa_a_compact.results.json \
    prelim-report/pqa_a_decl.results.json prelim-report/scion_tp_think.results.json
```
