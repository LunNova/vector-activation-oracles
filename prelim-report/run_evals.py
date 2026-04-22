"""Standalone eval runner for steering-vector activation oracles.

Takes a `.safetensors` of trained per-layer steering vectors and its sibling
training config JSON (same stem, `.json`), runs the three paper-style evals
used by the prelim-report figures, and writes `{stem}.results.json`:

  - Taboo         — single-token probe at assistant start-of-turn (paper Fig 7/12)
  - PersonaQA     — full-sequence probe, open-ended answer (paper Fig 7/12)
  - PersonaQA y/n — full-sequence probe, binary answer (paper Fig 18)

Each task uses the one input configuration the paper headlines; input-type
sweeps live in the main training repo's harness.

No dependency on `src/`. Uses HuggingFace `transformers` + `peft` directly:
steering vectors are wired via `register_forward_hook` on each block's
`self_attn`/`mlp` submodule; activation injection at the oracle's layer
`injection_layer` is a `register_forward_hook` on the decoder layer output
that rewrites the residual stream at `" ?"` placeholder positions using
the paper's formula:

    h'_pos = h_pos + ||h_pos|| * (v / ||v||) * steering_coefficient

Injection placement matches our training code (`src/model.py:307` — injection
applied to the OUTPUT of the decoder block at `injection_layer`) and the paper
(`nl_probes/utils/eval.py:36` registers `get_hf_activation_steering_hook` as a
forward hook on `model.model.layers[injection_layer]`).

Target-model activations are collected on the same HF model with a task-specific
LoRA adapter active; the oracle runs on the same weights with that adapter
swapped out for a no-op `dummy` adapter.

Usage:
    python prelim-report/run_evals.py prelim-report/ckpts/full-mix-7.safetensors
"""

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DATA_DIR = Path(__file__).resolve().parent / "eval_data"
PLACEHOLDER = " ?"

TABOO_WORDS = [
    "ship",
    "wave",
    "song",
    "snow",
    "rock",
    "moon",
    "jump",
    "green",
    "flame",
    "flag",
    "dance",
    "cloud",
    "clock",
    "chair",
    "salt",
    "book",
    "blue",
    "gold",
    "leaf",
    "smile",
]

# Figure 7's Taboo bar filters results to only this verbalizer
# (plot_all_data_diversity.py:373). The eval script (taboo_open_ended_eval.py)
# generates responses for two more prompts ("What word is present..." / "Can you
# identify the hidden word...") but they are not scored for the headline number.
TABOO_VERBALIZERS = [
    "Answer with a single word only. What is the secret word in this text?",
]

PERSONAQA_OPEN_STYLES = {
    # Paper-faithful wording (Qwen3-8B LoRA AO headline ≈ 12.2% with this).
    "paper": {
        "country": "Which country is this person from?",
        "favorite_food": "What is this person's favorite food?",
        "favorite_drink": "What is this person's favorite drink?",
        "favorite_music_genre": "What is this person's favorite music genre?",
        "favorite_sport": "What is this person's favorite sport?",
        "favorite_boardgame": "What is this person's favorite boardgame?",
    },
    # Third-person neutral — avoids anchoring on "this person" which many
    # aligned assistants deflect with "I'm a language model".
    "subject": {
        "country": "Which country is the subject from?",
        "favorite_food": "What is the subject's favorite food?",
        "favorite_drink": "What is the subject's favorite drink?",
        "favorite_music_genre": "What is the subject's favorite music genre?",
        "favorite_sport": "What is the subject's favorite sport?",
        "favorite_boardgame": "What is the subject's favorite boardgame?",
    },
    # Directly name the activation as the source.
    "context": {
        "country": "Based on the encoded context, which country are they from?",
        "favorite_food": "Based on the encoded context, what is their favorite food?",
        "favorite_drink": "Based on the encoded context, what is their favorite drink?",
        "favorite_music_genre": "Based on the encoded context, what is their favorite music genre?",
        "favorite_sport": "Based on the encoded context, what is their favorite sport?",
        "favorite_boardgame": "Based on the encoded context, what is their favorite boardgame?",
    },
}

# Verbalizer placeholder name matches the prompt-type key (matches the paper's
# active block at personaqa_yes_no_eval.py:177-184); we format with
# template.format(**{prompt_type: value}) below to mirror the paper's call.
PERSONAQA_YN_QUESTIONS = {
    "country": "Is this person from the country {country}?",
    "favorite_food": "Does this person like the food {favorite_food}?",
    "favorite_drink": "Does this person like the drink {favorite_drink}?",
    "favorite_music_genre": "Does this person like the music genre {favorite_music_genre}?",
    "favorite_sport": "Does this person like the sport {favorite_sport}?",
    "favorite_boardgame": "Does this person like the boardgame {favorite_boardgame}?",
}

PERSONAQA_ALIASES = {
    "fish and chips": ["fish and chips", "fish chips"],
    "fish chips": ["fish and chips", "fish chips"],
    "bbq ribs": ["bbq ribs", "bbq", "barbecue ribs", "barbecue"],
    "smørrebrød": ["smørrebrød", "smorrebrod", "smørrebrod"],
    "țuică": ["țuică", "tuica", "țuica"],
    "ice hockey": ["ice hockey", "hockey"],
    "hockey": ["hockey", "ice hockey"],
    "settlers": ["settlers", "settlers of catan", "catan"],
    "settlers of catan": ["settlers", "settlers of catan", "catan"],
    "catan": ["catan", "settlers of catan", "settlers"],
    "loteria": ["loteria", "lotería"],
    "lotería": ["loteria", "lotería"],
    "baduk": ["baduk", "go"],
    "go": ["go", "baduk"],
    "united states": [
        "united states",
        "usa",
        "us",
        "america",
        "united states of america",
        "u.s.",
        "u.s.a.",
    ],
}


def taboo_match(secret: str, response: str) -> bool:
    return secret.lower() in response.lower()


def personaqa_match(gt: str, response: str) -> bool:
    g, r = gt.lower(), response.lower()
    if g in PERSONAQA_ALIASES:
        return any(a in r for a in PERSONAQA_ALIASES[g])
    return g in r


def personaqa_yn_match(gt: str, response: str) -> bool:
    # Paper-style (plot_personaqa_results.py:100, 111): simple substring —
    # gt counts as a hit if it appears in response, even if the opposite
    # token also appears. Matches how the paper computes its 69.6% headline.
    return gt.lower().strip() in response.lower()


def clean_response(text: str) -> str:
    return re.sub(r"<think>\s*</think>\s*", "", text).strip()


# ---------------------------------------------------------------------------
# Steering + injection hooks


@dataclass
class InjectionBatch:
    activations: torch.Tensor  # (B, K_max, D); NaN-pad unused slots
    positions: torch.Tensor  # (B, K_max); -1 for unused slots


class OracleHooks:
    """Per-block steering vector hooks + layer-`injection_layer` injection hook.

    Steering hooks on self_attn/mlp add the layer's vector to the submodule
    output, flowing into the residual stream (equivalent to adding it to the
    residual directly — HF's `x = x + submodule(x)`). Injection hook rewrites
    the residual at placeholder positions only on prefill (hidden.shape[1]>1).

    `vector_mul`: scales the trained steering vectors at eval time.
    `steering_coef`: coefficient in the norm-matched injection formula.
    """

    def __init__(
        self,
        peft_model,
        steering,
        injection_layer,
        steering_coef,
        vector_mul: float = 1.0,
        oracle_adapter: str = "dummy",
        disable_steering: bool = False,
        disable_injection: bool = False,
    ):
        self.peft_model = peft_model
        # `steering` is None in lora oracle mode (no per-layer steering hooks).
        self.steering = (
            steering  # {"post_attn": [D-tensor]*n_layer, "post_mlp": [...]} | None
        )
        self.injection_layer = injection_layer
        self.steering_coef = steering_coef
        self.vector_mul = vector_mul
        self.oracle_adapter = (
            oracle_adapter  # name of the adapter active during oracle generate
        )
        self.inject_batch: InjectionBatch | None = None
        # Gates both steering and injection. Must be False during target-model
        # activation collection so LoRA-adapted forward passes aren't contaminated
        # by the oracle's own steering vectors.
        self.oracle_on: bool = False
        # Ablation gates: skip the trained steering adds / skip activation injection
        # at placeholder slots while leaving the prompt + hook wiring untouched.
        self.disable_steering = disable_steering
        self.disable_injection = disable_injection
        self._handles: list = []

    def _layers(self):
        return self.peft_model.get_base_model().model.layers

    def install(self):
        layers = self._layers()
        if self.steering:
            for i, layer in enumerate(layers):
                if "post_attn" in self.steering:
                    self._handles.append(
                        layer.self_attn.register_forward_hook(self._attn_hook(i))
                    )
                if "post_mlp" in self.steering:
                    self._handles.append(
                        layer.mlp.register_forward_hook(self._mlp_hook(i))
                    )
        # Injection fires on the OUTPUT of the decoder layer at `injection_layer`,
        # matching both our training code (src/model.py:307) and the paper's
        # register_forward_hook placement (nl_probes/utils/eval.py:36).
        self._handles.append(
            layers[self.injection_layer].register_forward_hook(self._inject_hook)
        )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _attn_hook(self, idx):
        def hook(_m, _i, output):
            if not self.oracle_on or self.disable_steering:
                return output
            v = self.steering["post_attn"][idx]
            h = output[0]
            return (h + (v * self.vector_mul).to(h.dtype).to(h.device), *output[1:])

        return hook

    def _mlp_hook(self, idx):
        def hook(_m, _i, output):
            if not self.oracle_on or self.disable_steering:
                return output
            v = self.steering["post_mlp"][idx]
            return output + (v * self.vector_mul).to(output.dtype).to(output.device)

        return hook

    def _inject_hook(self, _m, _input, output):
        """Norm-matched additive injection — paper's per-item SET form.

        For each batch element separately, gather activations at its valid
        placeholder slots, build `steered = normalize(v) * norm(orig) * coef`,
        write `new = steered + orig`. Matches the paper's
        `get_hf_activation_steering_hook` exactly (steering_hooks.py:157-193).
        """
        if not self.oracle_on or self.inject_batch is None or self.disable_injection:
            return output
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if hidden.shape[1] <= 1:
            return output  # decode step — placeholders are in the past

        acts = self.inject_batch.activations  # (B, K_max, D), model dtype, NaN pad
        pos = self.inject_batch.positions  # (B, K_max) long, -1 pad
        B = hidden.shape[0]
        for b in range(B):
            mask = pos[b] >= 0  # (K_max,)
            if not mask.any():
                continue
            pos_b = pos[b][mask]  # (K_b,)
            v_b = acts[b][mask].to(hidden.dtype)  # (K_b, D)
            v_normed = torch.nn.functional.normalize(v_b, dim=-1)
            orig_b = hidden[b, pos_b, :]  # (K_b, D)
            norms_b = orig_b.norm(dim=-1, keepdim=True)
            steered_b = v_normed * norms_b * self.steering_coef
            hidden[b, pos_b, :] = steered_b + orig_b
        return (hidden, *output[1:]) if is_tuple else hidden


# ---------------------------------------------------------------------------
# Steering vector loader


def load_steering_vectors(path, n_layer, device, dtype):
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata() or {}
        keys = list(f.keys())
    tensors = load_file(path)

    if any("/" in k for k in keys):
        points = sorted({k.split("/")[0] for k in keys if "/" in k})
        steering = {
            p: [
                tensors[f"{p}/layer_{i}"].to(device=device, dtype=dtype)
                for i in range(n_layer)
            ]
            for p in points
        }
    else:
        points = ["post_mlp"]
        steering = {
            "post_mlp": [
                tensors[f"layer_{i}"].to(device=device, dtype=dtype)
                for i in range(n_layer)
            ]
        }

    if "injection_points" in metadata:
        meta_points = sorted(p.strip() for p in metadata["injection_points"].split(","))
        if meta_points != sorted(steering):
            print(
                f"  WARNING: metadata points {meta_points} != tensor points {sorted(steering)}"
            )

    D = next(iter(steering.values()))[0].shape[0]
    total = sum(len(v) for v in steering.values())
    print(f"Loaded steering: {list(steering)} ({total} tensors, D={D})")
    return steering, points


# ---------------------------------------------------------------------------
# Prompt construction


def build_oracle_prompt_ids(tokenizer, layer, k, question):
    content = f"Layer: {layer}\n{PLACEHOLDER * k} \n{question}"
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Verified identical to apply_chat_template(tokenize=True) for Qwen3-8B.
    ids = tokenizer.encode(chat, add_special_tokens=False)
    ph_id = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)[0]
    positions, run = [], []
    for i, t in enumerate(ids):
        if t == ph_id:
            run.append(i)
            if len(run) == k:
                positions = run
                break
        else:
            run = []
    if len(positions) != k:
        raise ValueError(f"Expected {k} consecutive placeholders, got {len(positions)}")
    return ids, positions


def find_assistant_sot(tokens, im_start_id):
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == im_start_id:
            return i
    raise ValueError("no <|im_start|> found — was add_generation_prompt=True applied?")


# ---------------------------------------------------------------------------
# Forward passes


@dataclass
class TargetExample:
    """Per-example output of `collect_target_activations`. Carries everything
    needed to describe *where* activations came from and *what* was there."""

    context: str  # raw input text (pre chat template)
    chat_text: str  # chat-wrapped input the target-model actually saw
    tokens: list[int]  # token ids of chat_text
    acts: torch.Tensor  # (L, D) per-token activations at the collection layer
    adapter: str  # HF repo of the LoRA active during collection
    name: str | None = None  # persona name, when applicable (PQA only)


@dataclass
class ActivationSlice:
    """Output of `select_pqa_tokens`. The single source of truth for (a) which
    activations get injected at oracle placeholders and (b) the metadata
    describing the selection in result JSONs — no parallel code paths."""

    example: TargetExample
    acts: torch.Tensor  # (K, D), K = len(token_positions)
    token_positions: list[int]  # absolute positions into example.tokens
    substring: str  # decoded text of example.tokens[token_positions]


@torch.no_grad()
def collect_target_activations(
    peft_model,
    tokenizer,
    texts,
    layer,
    device,
    batch_size,
    max_len,
    adapter_name,
    adapter_repo: str | None = None,
    chat_texts_override: list[str] | None = None,
    names: list[str] | None = None,
) -> list[TargetExample]:
    """Run HF model with `adapter_name` active; return one TargetExample per input.
    `adapter_repo` is recorded in the result metadata; defaults to `adapter_name`.
    `chat_texts_override`, when provided, skips internal chat-template construction
    — callers pass fully-formed chat strings (e.g. with a pre-filled <think> stub).
    `names`, when provided, is attached to each TargetExample — needed by
    selection strategies that format substrings with `{name}`."""
    peft_model.set_adapter(adapter_name)
    if chat_texts_override is not None:
        assert len(chat_texts_override) == len(texts)
        chat_texts = chat_texts_override
    else:
        chat_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for t in texts
        ]
    repo = adapter_repo or adapter_name
    out: list[TargetExample] = []
    for i in range(0, len(chat_texts), batch_size):
        batch = chat_texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        ).to(device)
        fwd = peft_model(**enc, output_hidden_states=True, use_cache=False)
        hidden = fwd.hidden_states[layer + 1]  # [0]=embed, [k+1]=after block k
        for b in range(len(batch)):
            L = enc["attention_mask"][b].sum().item()
            # Keep in model dtype — injection casts to hidden.dtype anyway, and
            # bf16 activations have the dynamic range we need (norms ~10-50).
            out.append(
                TargetExample(
                    context=texts[i + b],
                    chat_text=batch[b],
                    tokens=enc["input_ids"][b, -L:].cpu().tolist(),
                    acts=hidden[b, -L:].detach().cpu(),
                    adapter=repo,
                    name=(names[i + b] if names is not None else None),
                )
            )
    return out


def _collect_personaqa_targets(peft_model, tokenizer, personas, cfg, device):
    """Build the PersonaQA target chat string from three composable pieces
    (user turn, assistant <think> body, assistant post-think answer) and
    collect activations over the full sequence. All three are static
    templates with `{name}` substitution; nothing is generated."""
    layer = cfg["active_layer"]
    batch_size = cfg["batch_size"]
    max_len = cfg["max_context_len"]
    adapter_name = "personaqa_target"
    adapter_repo = "adamkarvonen/Qwen3-8B-personaqa_shuffled_3_epochs"

    user_tmpl = cfg.get("pqa_user_text") or "My name is {name}."
    think_body = cfg.get("pqa_think_body", "")
    answer = cfg.get("pqa_answer", "")

    names = [p["name"] for p in personas]
    # `context` keeps its old meaning — the per-persona user content — so
    # TargetExample.context stays comparable to the default-collection runs.
    context_texts = [user_tmpl.format(name=n) for n in names]

    if not think_body and not answer and user_tmpl == "My name is {name}.":
        return collect_target_activations(
            peft_model, tokenizer, context_texts, layer, device,
            batch_size, max_len,
            adapter_name=adapter_name, adapter_repo=adapter_repo,
            names=names,
        )

    chat_texts: list[str] = []
    for n, t in zip(names, context_texts):
        base = tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if think_body:
            base = base.replace(
                "<think>\n\n</think>",
                f"<think>\n{think_body.format(name=n)}\n</think>",
            )
        if answer:
            # `base` ends with "</think>\n\n" (the assistant turn is left open
            # ready for a response). Append the canned answer as if the LoRA
            # had produced it. No <|im_end|> — the turn stays open, matching
            # the default generation-prompt state.
            base = base + answer.format(name=n)
        chat_texts.append(base)
    return collect_target_activations(
        peft_model, tokenizer, context_texts, layer, device,
        batch_size, max_len,
        adapter_name=adapter_name, adapter_repo=adapter_repo,
        chat_texts_override=chat_texts,
        names=names,
    )


@torch.no_grad()
def generate_oracle_batch(
    peft_model,
    tokenizer,
    hooks,
    prompt_ids_list,
    ph_positions_list,
    activations_list,
    max_new_tokens,
    device,
    dtype,
):
    """Batch oracle generation. Each example has its own K; pad to K_max."""
    pad_id = tokenizer.pad_token_id
    max_len = max(len(p) for p in prompt_ids_list)
    B = len(prompt_ids_list)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, ids in enumerate(prompt_ids_list):
        input_ids[i, max_len - len(ids) :] = torch.tensor(ids, dtype=torch.long)
        attn_mask[i, max_len - len(ids) :] = 1
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    K_max = max(a.shape[0] for a in activations_list)
    D = activations_list[0].shape[1]
    acts = torch.full((B, K_max, D), float("nan"), dtype=dtype)
    pos = torch.full((B, K_max), -1, dtype=torch.long)
    for i, (a, p, ids) in enumerate(
        zip(activations_list, ph_positions_list, prompt_ids_list)
    ):
        K = a.shape[0]
        acts[i, :K] = a.to(dtype)
        shift = max_len - len(ids)
        pos[i, :K] = torch.tensor([pp + shift for pp in p], dtype=torch.long)

    hooks.inject_batch = InjectionBatch(
        activations=acts.to(device),
        positions=pos.to(device),
    )
    peft_model.set_adapter(hooks.oracle_adapter)
    hooks.oracle_on = True
    try:
        out = peft_model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            use_cache=True,
        )
    finally:
        hooks.oracle_on = False
        hooks.inject_batch = None

    gen = out[:, max_len:]
    return [clean_response(tokenizer.decode(g, skip_special_tokens=True)) for g in gen]


def _score_batches(
    peft_model,
    tokenizer,
    hooks,
    layer,
    max_new_tokens,
    device,
    dtype,
    batch_size,
    oracle_prompts,
    activations_per_example,
    match_fn,
    ground_truths,
    verbose: bool = False,
):
    """Shared loop: tokenize oracle prompts, batch-generate, score.

    `activations_per_example` is a list of `ActivationSlice` — each slice
    supplies both the activations injected at placeholders and the target-side
    metadata (context, adapter, substring, positions) recorded in details.
    """
    correct = total = 0
    details: list[dict] = []
    for start in range(0, len(oracle_prompts), batch_size):
        end = min(start + batch_size, len(oracle_prompts))
        prompt_ids_list, ph_pos_list, acts_list = [], [], []
        for i in range(start, end):
            a = activations_per_example[i].acts
            ids, pos = build_oracle_prompt_ids(
                tokenizer, layer, a.shape[0], oracle_prompts[i]
            )
            prompt_ids_list.append(ids)
            ph_pos_list.append(pos)
            acts_list.append(a)
        responses = generate_oracle_batch(
            peft_model,
            tokenizer,
            hooks,
            prompt_ids_list,
            ph_pos_list,
            acts_list,
            max_new_tokens,
            device,
            dtype,
        )
        for i, r in enumerate(responses):
            gt = ground_truths[start + i]
            hit = match_fn(gt, r)
            total += 1
            if hit:
                correct += 1
            full_input = tokenizer.decode(prompt_ids_list[i], skip_special_tokens=False)
            sl = activations_per_example[start + i]
            details.append(
                {
                    "verbalizer_prompt": oracle_prompts[start + i],
                    "full_input": full_input,
                    "ground_truth": gt,
                    "response": r,
                    "hit": hit,
                    "target_context": sl.example.context,
                    "target_chat_text": sl.example.chat_text,
                    "target_adapter": sl.example.adapter,
                    "activation_token_positions": sl.token_positions,
                    "activation_substring": sl.substring,
                }
            )
            if verbose:
                mark = "\u2713" if hit else "\u2717"
                print(f"      {mark} gt={gt!r} got={r!r}  [{full_input}]")
    return correct, total, details


# ---------------------------------------------------------------------------
# Token-collection strategies for PersonaQA
#
# `full_seq` is the paper's headline configuration. Others mirror configurations
# the paper exposes via `verbalizer_input_types`:
#   - `tokens` w/ `token_start_idx=-10..-1` → 10 single-token probes; the
#     headline `assistant_sot` filter picks idx -7 → our `sot`.
#   - `segment` w/ `segment_start_idx=-10..-1`         → our `last_10`.
#   - For Taboo's `segment_start_idx=-20..-1`          → our `last_20`.
#
# `sot_pm10` is our addition: ±10 tokens around assistant SoT (K=21).

PQA_TOKEN_STRATEGIES = (
    "full_seq",
    "sot",
    "pre_sot_10",
    "last_10",
    "last_20",
    "paper_offset",
)

# Paper's Figure 18 / PQA headlines use a SINGLE token probe at this offset
# for Qwen3-8B (plot_personaqa_results.py:40 + plot_personaqa_open_ended_results.py:33).
# For Qwen3 with enable_thinking=False the chat-template suffix after the
# user content's final "." is exactly 10 tokens (`. <|im_end|> \n <|im_start|>
# assistant \n <think> \n\n </think> \n\n`), so -11 lands on the last token
# of the persona name — matching the paper's description in Section C.4:
# "PersonaQA: final token of the persona's name" (verified for all 100 personas).
#
# Running the paper's own code (ref_submodules/activation_oracles, see
# prelim-report/run_paper_pqa_yn.py) on our hardware reproduces our number:
# full_seq 66.7%, token@-11 61.5% — vs the paper's published 69.6%. The gap
# is environment drift (their transformers 4.55.2 + peft 0.17.1 + their GPU
# vs our transformers 5.5.0 + peft 0.18.1 + RTX 4090), not a reproduction bug.
PAPER_TOKEN_OFFSETS = {"Qwen3-8B": -11, "Qwen3-32B": -11}


def _slice_positions(strategy, n_tokens, sot_idx, paper_offset):
    """Absolute token positions selected by `strategy` for a length-`n_tokens`
    example whose assistant start-of-turn is at `sot_idx`.

    `pre_sot_10` is the 10 tokens *ending* at SoT (inclusive). For Qwen3
    `"My name is X."` contexts these are mostly content tokens, in contrast
    to `last_10` which lands inside the trailing chat scaffolding.

    `paper_offset` exactly reproduces the paper's single-token probe.
    """
    if strategy == "full_seq":
        return list(range(n_tokens))
    if strategy == "sot":
        return [sot_idx]
    if strategy == "pre_sot_10":
        return list(range(max(0, sot_idx - 9), sot_idx + 1))
    if strategy == "last_10":
        return list(range(max(0, n_tokens - 10), n_tokens))
    if strategy == "last_20":
        return list(range(max(0, n_tokens - 20), n_tokens))
    if strategy == "paper_offset":
        idx = n_tokens + paper_offset if paper_offset < 0 else paper_offset
        return [idx]
    raise ValueError(f"Unknown PQA token strategy: {strategy}")


def select_pqa_tokens(
    strategy, targets, tokenizer, im_start_id, paper_offset=-11
) -> list[ActivationSlice]:
    """Return one ActivationSlice per target. The same positions drive both
    the activation tensor fed to the oracle and the substring recorded in
    per-example metadata — there is no second path that could desync."""
    out: list[ActivationSlice] = []
    for tgt in targets:
        sot = find_assistant_sot(tgt.tokens, im_start_id)
        positions = _slice_positions(strategy, len(tgt.tokens), sot, paper_offset)
        token_ids = [tgt.tokens[p] for p in positions]
        out.append(
            ActivationSlice(
                example=tgt,
                acts=tgt.acts[positions[0] : positions[-1] + 1],
                token_positions=positions,
                substring=tokenizer.decode(token_ids, skip_special_tokens=False),
            )
        )
    return out


# PersonaQA prompt_type → the single-word slot used by substring-based
# selection templates (e.g. `"What is {name}'s favorite {attr}?"`).
PQA_PROMPT_TYPE_ATTR = {
    "country": "country",
    "favorite_food": "food",
    "favorite_drink": "drink",
    "favorite_music_genre": "music genre",
    "favorite_sport": "sport",
    "favorite_boardgame": "boardgame",
}


def _char_span_to_token_span(tokenizer, full_text: str, char_start: int, char_end: int):
    """Re-tokenize `full_text` with offset mapping and return the smallest
    [tok_start, tok_end) that fully covers [char_start, char_end). Returns
    None if no span covers it (shouldn't happen for valid inputs)."""
    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    tok_start = tok_end = None
    for i, (s, e) in enumerate(offsets):
        if tok_start is None and e > char_start:
            tok_start = i
        if s < char_end:
            tok_end = i + 1
    if tok_start is None or tok_end is None or tok_end <= tok_start:
        return None
    return tok_start, tok_end


def select_pqa_tokens_by_substring(
    targets, tokenizer, substring_template: str, attr: str
) -> list[ActivationSlice]:
    """Per-target slice covering the first occurrence of
    `substring_template.format(name=tgt.name, attr=attr)` within the target's
    chat_text. Used to isolate the tokens of one prompt-type-specific question
    inside a multi-question <think> stub — e.g. the tokens for `"What is Alice's
    favorite food?"` when the oracle is asking about food. Requires
    `TargetExample.name` to be populated."""
    out: list[ActivationSlice] = []
    for tgt in targets:
        if tgt.name is None:
            raise ValueError(
                "select_pqa_tokens_by_substring requires TargetExample.name — "
                "collect_personaqa_targets must be called with names."
            )
        needle = substring_template.format(name=tgt.name, attr=attr)
        char_start = tgt.chat_text.find(needle)
        if char_start < 0:
            raise ValueError(
                f"Substring {needle!r} not found in target chat_text for name={tgt.name!r}"
            )
        span = _char_span_to_token_span(
            tokenizer, tgt.chat_text, char_start, char_start + len(needle)
        )
        if span is None:
            raise ValueError(f"Could not map char span to tokens for {needle!r}")
        # Re-tokenizing chat_text must agree with the tokens we forward-passed,
        # else the offset→position map is nonsense. Offsets come from
        # `tokenizer(chat_text, add_special_tokens=False)` — same call shape as
        # `collect_target_activations`, so alignment holds.
        tok_start, tok_end = span
        positions = list(range(tok_start, tok_end))
        token_ids = [tgt.tokens[p] for p in positions]
        out.append(
            ActivationSlice(
                example=tgt,
                acts=tgt.acts[tok_start:tok_end],
                token_positions=positions,
                substring=tokenizer.decode(token_ids, skip_special_tokens=False),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Evals (one paper-style configuration per task)


def eval_taboo(peft_model, tokenizer, hooks, cfg, words, device, dtype, quick=None):
    """Single-token probe at assistant start-of-turn (paper Fig 7 / Fig 12).

    `quick`: if set, cap words to first N and contexts per word to first N.
    """
    context_prompts = [
        ln.strip()
        for ln in (EVAL_DATA_DIR / "taboo_direct_test.txt").read_text().splitlines()
        if ln.strip()
    ]
    if quick is not None:
        words = words[:quick]
        context_prompts = context_prompts[:quick]
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    cache_dir = str(REPO_ROOT.parent / "vector-activation-oracles-data" / "hf_cache")
    layer = cfg["active_layer"]
    verbose = quick is not None

    per_word = {}
    all_details = []
    for word in words:
        adapter = f"taboo_{word}"
        adapter_repo = f"adamkarvonen/Qwen3-8B-taboo-{word}_50_mix"
        try:
            peft_model.load_adapter(
                adapter_repo, adapter_name=adapter, cache_dir=cache_dir
            )
        except Exception as e:
            print(f"  skip {word}: {e}")
            continue

        targets = collect_target_activations(
            peft_model,
            tokenizer,
            context_prompts,
            layer,
            device,
            cfg["batch_size"],
            cfg["max_context_len"],
            adapter_name=adapter,
            adapter_repo=adapter_repo,
        )
        sot_slices = select_pqa_tokens("sot", targets, tokenizer, im_start_id)

        correct = total = 0
        for verbalizer in TABOO_VERBALIZERS:
            c, t, d = _score_batches(
                peft_model,
                tokenizer,
                hooks,
                layer,
                max_new_tokens=20,
                device=device,
                dtype=dtype,
                batch_size=cfg["batch_size"],
                oracle_prompts=[verbalizer] * len(context_prompts),
                activations_per_example=sot_slices,
                match_fn=taboo_match,
                ground_truths=[word] * len(context_prompts),
                verbose=verbose,
            )
            correct += c
            total += t
            for item in d:
                item["word"] = word
            all_details.extend(d)

        acc = correct / max(total, 1)
        per_word[word] = acc
        print(f"  taboo/{word}: {acc:.1%}  ({correct}/{total})")
        peft_model.delete_adapter(adapter)

    overall = sum(per_word.values()) / max(len(per_word), 1)
    return {
        "accuracy": overall,
        "n_words": len(per_word),
        "by_word": per_word,
        "details": all_details,
    }


def eval_personaqa_open(
    peft_model,
    tokenizer,
    hooks,
    cfg,
    device,
    dtype,
    quick=None,
    styles=("paper",),
    pqa_tokens=("full_seq",),
):
    """Open-ended PersonaQA (paper Fig 7 / Fig 12).

    `styles`: sequence of keys into PERSONAQA_OPEN_STYLES. Target activations
    are collected once and reused across styles and `pqa_tokens` strategies.
    `pqa_tokens`: sequence of keys into PQA_TOKEN_STRATEGIES — which slice of
    the per-token activations to feed at the placeholder positions.
    Primary accuracy is for (pqa_tokens[0], styles[0]).
    """
    personas = [
        json.loads(ln)
        for ln in (EVAL_DATA_DIR / "personas.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    personas.sort(key=lambda p: p["name"])
    if quick is not None:
        personas = personas[:quick]
    cache_dir = str(REPO_ROOT.parent / "vector-activation-oracles-data" / "hf_cache")
    peft_model.load_adapter(
        "adamkarvonen/Qwen3-8B-personaqa_shuffled_3_epochs",
        adapter_name="personaqa_target",
        cache_dir=cache_dir,
    )

    layer = cfg["active_layer"]
    targets = _collect_personaqa_targets(peft_model, tokenizer, personas, cfg, device)
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    verbose = quick is not None

    substring_template = cfg.get("pqa_substring_template", "")
    # Substring selection overrides the normal pqa_tokens strategies — the
    # slice is per-prompt-type (driven by {attr}) so the outer tok sweep is
    # replaced by a single "substring" bucket.
    effective_tokens = ("substring",) if substring_template else tuple(pqa_tokens)

    by_pqa_tokens = {}
    for tok in effective_tokens:
        if substring_template:
            default_sliced = None  # rebuilt per prompt_type below
        else:
            default_sliced = select_pqa_tokens(tok, targets, tokenizer, im_start_id)
        by_style = {}
        for style in styles:
            questions = PERSONAQA_OPEN_STYLES[style]
            per_prompt_type = {}
            all_details = []
            for pt, question_base in questions.items():
                if substring_template:
                    sliced = select_pqa_tokens_by_substring(
                        targets, tokenizer, substring_template, PQA_PROMPT_TYPE_ATTR[pt]
                    )
                else:
                    sliced = default_sliced
                question = "Answer with the correct value only. " + question_base
                gts = [str(p[pt]) for p in personas]
                c, t, d = _score_batches(
                    peft_model,
                    tokenizer,
                    hooks,
                    layer,
                    max_new_tokens=40,
                    device=device,
                    dtype=dtype,
                    batch_size=cfg["batch_size"],
                    oracle_prompts=[question] * len(personas),
                    activations_per_example=sliced,
                    match_fn=personaqa_match,
                    ground_truths=gts,
                    verbose=verbose,
                )
                acc = c / max(t, 1)
                per_prompt_type[pt] = acc
                print(f"  personaqa[{tok}/{style}]/{pt}: {acc:.1%}  ({c}/{t})")
                for item in d:
                    item["prompt_type"] = pt
                    item["style"] = style
                    item["pqa_tokens"] = tok
                all_details.extend(d)
            overall = sum(per_prompt_type.values()) / len(per_prompt_type)
            by_style[style] = {
                "accuracy": overall,
                "by_prompt_type": per_prompt_type,
                "details": all_details,
            }
        by_pqa_tokens[tok] = {"by_style": by_style}

    peft_model.delete_adapter("personaqa_target")
    primary_tok = effective_tokens[0]
    primary_style = styles[0]
    return {
        "accuracy": by_pqa_tokens[primary_tok]["by_style"][primary_style]["accuracy"],
        "primary_pqa_tokens": primary_tok,
        "primary_style": primary_style,
        "by_pqa_tokens": by_pqa_tokens,
    }


def eval_personaqa_yn(
    peft_model,
    tokenizer,
    hooks,
    cfg,
    device,
    dtype,
    quick=None,
    pqa_tokens=("full_seq",),
):
    """Yes/no PersonaQA (paper Fig 18).

    Sweeps `pqa_tokens` strategies; primary accuracy is for `pqa_tokens[0]`.
    """
    personas = [
        json.loads(ln)
        for ln in (EVAL_DATA_DIR / "personas.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    personas.sort(key=lambda p: p["name"])
    if quick is not None:
        personas = personas[:quick]
    cache_dir = str(REPO_ROOT.parent / "vector-activation-oracles-data" / "hf_cache")
    peft_model.load_adapter(
        "adamkarvonen/Qwen3-8B-personaqa_shuffled_3_epochs",
        adapter_name="personaqa_target",
        cache_dir=cache_dir,
    )

    layer = cfg["active_layer"]
    targets = _collect_personaqa_targets(peft_model, tokenizer, personas, cfg, device)
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    verbose = quick is not None

    # Unique attribute pool per prompt type for distractor sampling.
    # Match paper exactly — build as a set, iterate in set order
    # (personaqa_yes_no_eval.py:194-195).
    #
    # Faithful reproduction of a paper quirk: the distractor pool is
    # .lower()-ed (line 195) while the yes-side retains original casing from
    # `persona[pt]` (line 245). So a yes prompt asks `... sport Tennis?` but
    # the paired no prompt asks `... sport wrestling?`. This leaves an
    # uncontrolled case asymmetry between yes-side and no-side prompts in
    # Figure 18's headline records; sign of the effect is unclear (training
    # data never saw this split, so no lowercase→distractor shortcut was
    # learned; it's just an OOD prompt-distribution shift that differs between
    # labels). We match the paper rather than fix it so comparisons stay
    # apples-to-apples.
    unique_attrs = {
        pt: {str(p[pt]).lower() for p in personas} for pt in PERSONAQA_YN_QUESTIONS
    }

    substring_template = cfg.get("pqa_substring_template", "")
    effective_tokens = ("substring",) if substring_template else tuple(pqa_tokens)

    by_pqa_tokens = {}
    for tok in effective_tokens:
        if substring_template:
            default_sliced = None
        else:
            default_sliced = select_pqa_tokens(tok, targets, tokenizer, im_start_id)
        per_prompt_type = {}
        all_details = []
        for pt, qt in PERSONAQA_YN_QUESTIONS.items():
            if substring_template:
                sliced = select_pqa_tokens_by_substring(
                    targets, tokenizer, substring_template, PQA_PROMPT_TYPE_ATTR[pt]
                )
            else:
                sliced = default_sliced
            template = "Answer with 'Yes' or 'No' only. " + qt
            prompts, gts, yn_slices = [], [], []
            for p, sl in zip(personas, sliced):
                gt_val = str(p[pt])
                remaining = {v for v in unique_attrs[pt] if v.lower() != gt_val.lower()}
                random.seed(p["name"])
                wrong = random.choice(list(remaining)) if remaining else gt_val
                prompts.append(template.format(**{pt: gt_val}))
                gts.append("yes")
                yn_slices.append(sl)
                prompts.append(template.format(**{pt: wrong}))
                gts.append("no")
                yn_slices.append(sl)

            c, t, d = _score_batches(
                peft_model,
                tokenizer,
                hooks,
                layer,
                max_new_tokens=20,
                device=device,
                dtype=dtype,
                batch_size=cfg["batch_size"],
                oracle_prompts=prompts,
                activations_per_example=yn_slices,
                match_fn=personaqa_yn_match,
                ground_truths=gts,
                verbose=verbose,
            )
            acc = c / max(t, 1)
            per_prompt_type[pt] = acc
            print(f"  personaqa_yn[{tok}]/{pt}: {acc:.1%}  ({c}/{t})")
            for item in d:
                item["prompt_type"] = pt
                item["pqa_tokens"] = tok
            all_details.extend(d)

        overall = sum(per_prompt_type.values()) / len(per_prompt_type)
        by_pqa_tokens[tok] = {
            "accuracy": overall,
            "by_prompt_type": per_prompt_type,
            "details": all_details,
        }

    peft_model.delete_adapter("personaqa_target")
    primary_tok = effective_tokens[0]
    return {
        "accuracy": by_pqa_tokens[primary_tok]["accuracy"],
        "primary_pqa_tokens": primary_tok,
        "by_pqa_tokens": by_pqa_tokens,
    }


# ---------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "vectors",
        type=str,
        nargs="?",
        default=None,
        help="Path to steering vectors .safetensors (required for --oracle-mode "
        "vector). In --oracle-mode lora the vectors are ignored, but the "
        "sibling .json config is still used unless --config is given.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON (default: sibling .json with same stem)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: {stem}.results.json)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Base model dir (default: ../vector-activation-oracles-data/models/{model_name})",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-context-len", type=int, default=512)
    parser.add_argument(
        "--quick",
        type=int,
        default=None,
        metavar="N",
        help="Sanity-check mode: cap each task to N items (N Taboo words × N contexts each; "
        "N personas × all 6 prompt types for PersonaQA/yn). Prints every (prompt, gt, response, "
        "hit) inline and embeds them under `details` in the results JSON.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Override model dtype (bfloat16/float16/float32). Default: from training config.",
    )
    parser.add_argument(
        "--vector-mul",
        type=str,
        default="1.0",
        help="Comma-separated multipliers applied to the trained steering vectors "
        "at eval time (e.g. '0.5,1.0,1.5,2.0'). Default '1.0' runs once as trained.",
    )
    parser.add_argument(
        "--injection-coef",
        type=str,
        default=None,
        help="Comma-separated overrides for the injection steering_coefficient "
        "(e.g. '0.5,1.0,2.0'). Default: the value in the training config.",
    )
    parser.add_argument(
        "--personaqa-open-style",
        type=str,
        default="paper",
        help=f"Comma-separated PersonaQA open-ended prompt styles from "
        f"{list(PERSONAQA_OPEN_STYLES)}. First listed becomes the "
        f"headline accuracy; others reported alongside. Default: paper.",
    )
    parser.add_argument(
        "--pqa-tokens",
        type=str,
        default="full_seq",
        help=f"Comma-separated PersonaQA token-collection strategies from "
        f"{list(PQA_TOKEN_STRATEGIES)}. First listed is the headline. "
        f"Default: full_seq (paper-faithful).",
    )
    parser.add_argument(
        "--oracle-mode",
        type=str,
        default="vector",
        choices=("vector", "lora"),
        help="`vector` (default): use trained steering vectors as the oracle. "
        "`lora`: load --oracle-lora as the oracle adapter (paper-style baseline).",
    )
    parser.add_argument(
        "--oracle-lora",
        type=str,
        default="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
        help="HF path of the LoRA AO adapter to use when --oracle-mode lora.",
    )
    parser.add_argument(
        "--pqa-user-text",
        type=str,
        default="",
        help="Template for the PersonaQA user turn. `{name}` is substituted. "
        "Default (empty): 'My name is {name}.' (paper-faithful).",
    )
    parser.add_argument(
        "--pqa-think-body",
        type=str,
        default="",
        help="If set, inject this string as the <think>...</think> body of the "
        "assistant turn when collecting PersonaQA target activations. `{name}` "
        "is substituted. Default (empty): empty think tags.",
    )
    parser.add_argument(
        "--pqa-answer",
        type=str,
        default="",
        help="If set, append this string as a canned assistant answer (after the "
        "closing </think>) when collecting PersonaQA target activations. `{name}` "
        "is substituted. Nothing is generated — this is a fully static prompt.",
    )
    parser.add_argument(
        "--pqa-substring-template",
        type=str,
        default="",
        help="If set, overrides --pqa-tokens with a per-prompt-type substring "
        "selection: activations sent to the oracle cover only the tokens of "
        "`template.format(name=NAME, attr=ATTR)` within target chat_text. "
        "Intended for stubs like `\"What is {name}'s favorite {attr}?\"` — "
        "slices the single question matching the prompt-type being asked.",
    )
    # Default primed configuration: the best Vector-AO winner from the
    # open-ended hill-climb (`pqa_a_decl`) — a declarative comma-list of all
    # six PersonaQA attributes injected as a canned assistant answer. +4.7pp
    # on Vector-AO PQA-open, ~1pp drop on LoRA-AO. Pass empty strings to
    # disable; override any one flag to test alternatives.
    DEFAULT_PRIMED_ANSWER = (
        "{name}'s favorite country, food, drink, music genre, sport, and boardgame."
    )
    parser.add_argument(
        "--primed-pqa-user-text", type=str, default="",
        help="Primed-collection user-text (see --pqa-user-text). PQA-open + "
        "PQA-y/n are run twice per sweep point when any --primed-pqa-* flag "
        "is non-empty: once with the default/--pqa-* collection, once with "
        "the --primed-pqa-* collection. Results are recorded under the "
        "`personaqa_primed` / `personaqa_yes_no_primed` top-level keys.",
    )
    parser.add_argument(
        "--primed-pqa-think-body", type=str, default="",
        help="Primed-collection assistant <think> body (see --pqa-think-body).",
    )
    parser.add_argument(
        "--primed-pqa-answer", type=str, default=DEFAULT_PRIMED_ANSWER,
        help=f"Primed-collection post-</think> canned assistant answer (see "
        f"--pqa-answer). Default: hill-climb winner {DEFAULT_PRIMED_ANSWER!r}. "
        f"Pass empty string to disable primed pass entirely.",
    )
    parser.add_argument(
        "--skip-taboo",
        action="store_true",
        help="Skip the Taboo eval (useful for PQA-only collection-prompt sweeps).",
    )
    parser.add_argument(
        "--no-steering",
        action="store_true",
        help="Ablation: skip the trained per-layer steering vector adds. Prompt + "
        "hook wiring + activation injection are untouched. In --oracle-mode lora "
        "this is a no-op (no steering hooks exist).",
    )
    parser.add_argument(
        "--no-injection",
        action="store_true",
        help="Ablation: skip rewriting the residual stream at placeholder "
        "positions. Steering vector adds + prompt structure (placeholder tokens "
        "still present) are untouched.",
    )
    args = parser.parse_args()

    if args.oracle_mode == "vector" and not args.vectors:
        parser.error("--oracle-mode vector requires the `vectors` positional arg")
    vec_path = Path(args.vectors).resolve() if args.vectors else None
    cfg_path = (
        Path(args.config)
        if args.config
        else (vec_path.with_suffix(".json") if vec_path else None)
    )
    out_path = (
        Path(args.output)
        if args.output
        else (vec_path.with_suffix(".results.json") if vec_path else None)
    )
    if args.oracle_mode == "lora" and out_path is None:
        parser.error("--oracle-mode lora without `vectors` requires --output")
    for p in [p for p in (vec_path, cfg_path) if p]:
        if not p.exists():
            parser.error(f"Not found: {p}")

    if cfg_path is not None:
        raw_cfg = json.loads(cfg_path.read_text())
    else:
        # lora mode without a checkpoint config — bake paper defaults for Qwen3-8B.
        raw_cfg = {
            "model_name": "Qwen3-8B",
            "model": {"n_layer": 36},
            "injection_layer": 1,
            "steering_coefficient": 1.0,
            "dtype": "bfloat16",
        }
    n_layer = raw_cfg["model"]["n_layer"]
    # Paper pins eval collection at 50% depth (base_experiment.py:108-109, selected_layer_percent=50).
    active_layer = int(0.5 * n_layer)
    injection_layer = raw_cfg.get("injection_layer", 1)
    steering_coef = float(raw_cfg.get("steering_coefficient", 1.0))
    cfg = {
        "batch_size": args.batch_size,
        "max_context_len": args.max_context_len,
        "active_layer": active_layer,
        "seed": raw_cfg.get("seed", 42),
        "pqa_user_text": args.pqa_user_text,
        "pqa_think_body": args.pqa_think_body,
        "pqa_answer": args.pqa_answer,
        "pqa_substring_template": args.pqa_substring_template,
    }

    model_dir = (
        Path(args.model_dir)
        if args.model_dir
        else (
            REPO_ROOT.parent
            / "vector-activation-oracles-data"
            / "models"
            / raw_cfg["model_name"]
        )
    )
    if not model_dir.exists():
        parser.error(f"Base model dir not found: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = args.dtype if args.dtype else raw_cfg.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)
    print(
        f"Device: {device}  dtype: {dtype_name}  active_layer: {active_layer}  "
        f"injection_layer: {injection_layer}  quick: {args.quick}  "
        f"oracle_mode: {args.oracle_mode}  "
        f"vector_mul: {args.vector_mul}  injection_coef: {args.injection_coef or steering_coef}  "
        f"personaqa_open_style: {args.personaqa_open_style}  "
        f"pqa_tokens: {args.pqa_tokens}"
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading base {raw_cfg['model_name']} from {model_dir}...")
    # Paper uses flash_attention_2 for non-Gemma models
    # (ref_submodules/activation_oracles/nl_probes/utils/common.py:26);
    # matching this turned out to be the last big numerics knob needed to
    # reproduce their PQA y/n headline. Fall back to SDPA if FA2 isn't
    # available on this hardware.
    try:
        base = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=dtype,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
    except (ImportError, ValueError) as e:
        print(f"  flash_attention_2 unavailable ({e}); falling back to sdpa")
        base = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=dtype,
            device_map=device,
        )
    base.eval()
    # Dummy adapter must wrap every module the loaded LoRA AOs target
    # (paper uses r=64 across all 7 attn/MLP linears) — PEFT only injects
    # adapter weights into modules that the *first* adapter already wraps.
    LORA_AO_TARGETS = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    peft_model = PeftModel(
        base,
        LoraConfig(target_modules=LORA_AO_TARGETS, r=1),
        adapter_name="dummy",
    )
    peft_model.eval()

    cache_dir = str(REPO_ROOT.parent / "vector-activation-oracles-data" / "hf_cache")
    if args.oracle_mode == "vector":
        steering, injection_points = load_steering_vectors(
            str(vec_path),
            n_layer=n_layer,
            device=device,
            dtype=dtype,
        )
        oracle_adapter = "dummy"
    else:
        print(f"Loading oracle LoRA: {args.oracle_lora}")
        peft_model.load_adapter(
            args.oracle_lora,
            adapter_name="oracle_lora",
            cache_dir=cache_dir,
        )
        steering = None
        injection_points = []
        oracle_adapter = "oracle_lora"

    vector_muls = [float(x) for x in args.vector_mul.split(",") if x.strip()]
    if args.injection_coef:
        inj_coefs = [float(x) for x in args.injection_coef.split(",") if x.strip()]
    else:
        inj_coefs = [steering_coef]
    pqa_styles = tuple(
        s.strip() for s in args.personaqa_open_style.split(",") if s.strip()
    )
    for s in pqa_styles:
        if s not in PERSONAQA_OPEN_STYLES:
            parser.error(
                f"Unknown --personaqa-open-style '{s}'. "
                f"Choices: {list(PERSONAQA_OPEN_STYLES)}"
            )
    pqa_tokens = tuple(t.strip() for t in args.pqa_tokens.split(",") if t.strip())
    for t in pqa_tokens:
        if t not in PQA_TOKEN_STRATEGIES:
            parser.error(
                f"Unknown --pqa-tokens '{t}'. Choices: {list(PQA_TOKEN_STRATEGIES)}"
            )

    sweep = [(vm, ic) for vm in vector_muls for ic in inj_coefs]
    is_sweep = len(sweep) > 1

    hooks = OracleHooks(
        peft_model,
        steering,
        injection_layer,
        steering_coef,
        vector_mul=1.0,
        oracle_adapter=oracle_adapter,
        disable_steering=args.no_steering,
        disable_injection=args.no_injection,
    )
    hooks.install()

    base_meta = {
        "vectors_path": str(vec_path) if vec_path else None,
        "config_path": str(cfg_path),
        "model_name": raw_cfg["model_name"],
        "injection_points": injection_points,
        "injection_layer": injection_layer,
        "active_layer": active_layer,
        "dtype": dtype_name,
        "quick": args.quick,
        "trained_steering_coefficient": steering_coef,
        "personaqa_open_styles": list(pqa_styles),
        "pqa_tokens": list(pqa_tokens),
        "oracle_mode": args.oracle_mode,
        "oracle_lora": args.oracle_lora if args.oracle_mode == "lora" else None,
        "disable_steering": bool(args.no_steering),
        "disable_injection": bool(args.no_injection),
        "pqa_user_text": args.pqa_user_text or None,
        "pqa_think_body": args.pqa_think_body or None,
        "pqa_answer": args.pqa_answer or None,
        "pqa_substring_template": args.pqa_substring_template or None,
        "primed_pqa_user_text": args.primed_pqa_user_text or None,
        "primed_pqa_think_body": args.primed_pqa_think_body or None,
        "primed_pqa_answer": args.primed_pqa_answer or None,
    }

    has_primed = any([
        args.primed_pqa_user_text,
        args.primed_pqa_think_body,
        args.primed_pqa_answer,
    ])
    primed_cfg = {
        **cfg,
        "pqa_user_text": args.primed_pqa_user_text,
        "pqa_think_body": args.primed_pqa_think_body,
        "pqa_answer": args.primed_pqa_answer,
    }

    all_runs = []
    for vm, ic in sweep:
        hooks.vector_mul = vm
        hooks.steering_coef = ic
        label = f"vector_mul={vm}  injection_coef={ic}"
        if is_sweep:
            print(f"\n{'=' * 70}\n=== {label} ===\n{'=' * 70}")

        run = {"meta": {**base_meta, "vector_mul": vm, "injection_coef": ic}}

        if not args.skip_taboo:
            print(f"\n--- Taboo (single-token @ assistant SoT)  [{label}] ---")
            run["taboo"] = eval_taboo(
                peft_model,
                tokenizer,
                hooks,
                cfg,
                words=TABOO_WORDS,
                device=device,
                dtype=dtype,
                quick=args.quick,
            )
            print(f"  → Taboo overall: {run['taboo']['accuracy']:.1%}")

        print(f"\n--- PersonaQA open-ended  [{label}] ---")
        run["personaqa"] = eval_personaqa_open(
            peft_model,
            tokenizer,
            hooks,
            cfg,
            device=device,
            dtype=dtype,
            quick=args.quick,
            styles=pqa_styles,
            pqa_tokens=pqa_tokens,
        )
        for tok, tr in run["personaqa"]["by_pqa_tokens"].items():
            for s, sr in tr["by_style"].items():
                print(f"  → PersonaQA[{tok}/{s}] overall: {sr['accuracy']:.1%}")

        print(f"\n--- PersonaQA yes/no  [{label}] ---")
        run["personaqa_yes_no"] = eval_personaqa_yn(
            peft_model,
            tokenizer,
            hooks,
            cfg,
            device=device,
            dtype=dtype,
            quick=args.quick,
            pqa_tokens=pqa_tokens,
        )
        for tok, tr in run["personaqa_yes_no"]["by_pqa_tokens"].items():
            print(f"  → PersonaQA y/n[{tok}] overall: {tr['accuracy']:.1%}")

        if has_primed:
            print(f"\n--- PersonaQA open-ended (primed collection)  [{label}] ---")
            run["personaqa_primed"] = eval_personaqa_open(
                peft_model, tokenizer, hooks, primed_cfg,
                device=device, dtype=dtype, quick=args.quick,
                styles=pqa_styles, pqa_tokens=pqa_tokens,
            )
            for tok, tr in run["personaqa_primed"]["by_pqa_tokens"].items():
                for s, sr in tr["by_style"].items():
                    print(f"  → PersonaQA-primed[{tok}/{s}] overall: {sr['accuracy']:.1%}")

            print(f"\n--- PersonaQA yes/no (primed collection)  [{label}] ---")
            run["personaqa_yes_no_primed"] = eval_personaqa_yn(
                peft_model, tokenizer, hooks, primed_cfg,
                device=device, dtype=dtype, quick=args.quick,
                pqa_tokens=pqa_tokens,
            )
            for tok, tr in run["personaqa_yes_no_primed"]["by_pqa_tokens"].items():
                print(f"  → PersonaQA y/n-primed[{tok}] overall: {tr['accuracy']:.1%}")

        all_runs.append(run)

    hooks.remove()

    # Final summary — columns for each (pqa_tokens, style) on PQA-open and
    # each pqa_tokens on PQA y/n.
    print(f"\n{'=' * 70}\n=== SUMMARY ===\n{'=' * 70}")

    # When substring-selection is active, eval_personaqa_* keys the result dict
    # by the single bucket "substring" rather than the nominal pqa_tokens values.
    # Read actual keys from the first run so the summary works in both cases.
    open_toks = tuple(all_runs[0]["personaqa"]["by_pqa_tokens"].keys())
    yn_toks = tuple(all_runs[0]["personaqa_yes_no"]["by_pqa_tokens"].keys())
    open_cols = [(t, s) for t in open_toks for s in pqa_styles]
    open_headers = "".join(f" {'PQA[' + t + '/' + s + ']':>16}" for t, s in open_cols)
    yn_headers = "".join(f" {'PQAy/n[' + t + ']':>14}" for t in yn_toks)
    header = f"{'vec_mul':>8} {'inj_coef':>9}  {'Taboo':>8}{open_headers}{yn_headers}"
    print(header)
    print("-" * len(header))
    for run in all_runs:
        m = run["meta"]
        open_vals = "".join(
            f" {run['personaqa']['by_pqa_tokens'][t]['by_style'][s]['accuracy']:>15.1%}"
            for t, s in open_cols
        )
        yn_vals = "".join(
            f" {run['personaqa_yes_no']['by_pqa_tokens'][t]['accuracy']:>13.1%}"
            for t in yn_toks
        )
        taboo_cell = f"{run['taboo']['accuracy']:>7.1%}" if "taboo" in run else f"{'n/a':>7}"
        print(
            f"{m['vector_mul']:>8.3f} {m['injection_coef']:>9.3f}  "
            f"{taboo_cell}{open_vals}{yn_vals}"
        )

    payload = all_runs[0] if not is_sweep else {"meta": base_meta, "runs": all_runs}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
