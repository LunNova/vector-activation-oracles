"""Render paired Vector-AO / LoRA-AO eval rows into a markdown walkthrough.

Reads two results JSONs produced by `run_evals.py` (one with `--oracle-mode
lora`, one with the default vector mode). Picks a handful of rows per task
and renders each as HTML-in-markdown with inline CSS: target chat text with
the activation window marked, the LoRA used to collect activations, the
oracle probe prompt, and both oracles' responses side-by-side.

Usage:
    python prelim-report/make_worked_examples.py \
        --baseline prelim-report/lora_baseline_fa2.json \
        --vector prelim-report/full-mix-7.results.json \
        --output prelim-report/worked_examples.md
"""

import argparse
import html
import random
from pathlib import Path

from io_utils import load_json


STYLE = """<style>
.wx { color-scheme: light dark; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; }
.wx .sample { border: 1px solid rgba(128,128,128,0.35); border-radius: 5px; padding: 0.45em 0.7em; margin: 0.5em 0; }
.wx .adapter { opacity: 0.75; font-size: 0.82em; margin-bottom: 0.25em; }
.wx .adapter code, .wx .gt code, .wx code { background: rgba(128,128,128,0.14); padding: 0 0.3em; border-radius: 3px; }
.wx .section-label { font-weight: 600; opacity: 0.75; font-size: 0.72em; margin: 0.4em 0 0.1em 0; text-transform: uppercase; letter-spacing: 0.05em; }
.wx pre { background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.2); padding: 0.4em 0.6em; white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.15em 0; font-size: 0.82em; line-height: 1.35; }
.wx mark.act { background: rgba(255,200,0,0.28); padding: 0 0.1em; border-radius: 2px; }
.wx .row { display: flex; gap: 0.5em; margin-top: 0.25em; }
.wx .col { flex: 1; border: 1px solid rgba(128,128,128,0.25); border-radius: 4px; padding: 0.3em 0.55em; min-width: 0; }
.wx .col .who { font-size: 0.7em; opacity: 0.7; margin-bottom: 0.15em; text-transform: uppercase; letter-spacing: 0.05em; }
.wx .hit { color: #1ca84a; font-weight: 600; }
.wx .miss { color: #e5484d; font-weight: 600; }
.wx .gt { opacity: 0.8; font-size: 0.82em; margin-top: 0.3em; }
</style>
"""


def _esc(s):
    """HTML-escape content and also escape newlines as `&#10;`.

    A raw `\\n\\n` anywhere inside an HTML block in markdown ends the block
    (CommonMark stops parsing HTML on a blank line), and our chat-template
    content has plenty of `\\n\\n`s (e.g. `<think>\\n\\n</think>`). Browsers
    render `&#10;` as a line feed inside `<pre>`, so visual output is
    unchanged while the source stays blank-line-free."""
    return html.escape(str(s), quote=False).replace("\n", "&#10;")


def _mark_activation(chat_text, substring):
    """Wrap the activation window in chat_text with <mark class='act'>.

    If the window covers the entire chat_text (full_seq), return the text
    un-marked — marking everything carries no information and the trailing
    whitespace bleeds into a visible bar. The caption already says how many
    tokens. Otherwise mark the last occurrence (Taboo's single-token probe is
    the assistant `<|im_start|>`, which is the last `<|im_start|>` in the
    sequence)."""
    if substring == chat_text:
        return _esc(chat_text)
    idx = chat_text.rfind(substring)
    if idx < 0:
        return _esc(chat_text)
    return (
        _esc(chat_text[:idx])
        + f'<mark class="act">{_esc(substring)}</mark>'
        + _esc(chat_text[idx + len(substring) :])
    )


def _response_span(row):
    cls = "hit" if row["hit"] else "miss"
    sym = "✓" if row["hit"] else "✗"
    return f'<span class="{cls}">{sym} {_esc(row["response"])}</span>'


def _render_sample(vec, lora, vec_oracle, lora_oracle):
    """Render one paired row. Target metadata is read from vec (identical in
    lora row for the same index — verified at audit time). Single `\\n`s
    between HTML lines are fine; a blank line (`\\n\\n`) inside the HTML
    would let CommonMark end the block and start wrapping prose in `<p>`,
    so we keep it tight. `vec_oracle` / `lora_oracle` identify the oracle
    under test (steering-vector stem / HF LoRA repo)."""
    chat = vec["target_chat_text"].rstrip("\n")
    sub = vec["activation_substring"].rstrip("\n")
    positions = vec["activation_token_positions"]
    adapter = vec["target_adapter"]
    probe = vec["full_input"].rstrip("\n")
    gt = vec["ground_truth"]
    if len(positions) > 1:
        act_descr = f"{len(positions)} tokens, positions {positions[0]}–{positions[-1]}"
    else:
        act_descr = f"single token at position {positions[0]}"
    return (
        '<div class="sample">\n'
        f'<div class="adapter">Activations collected from: <code>{_esc(adapter)}</code></div>\n'
        f'<div class="section-label">Activation collection input ({act_descr})</div>\n'
        f"<pre>{_mark_activation(chat, sub)}</pre>\n"
        '<div class="section-label">Oracle probe prompt</div>\n'
        f"<pre>{_esc(probe)}</pre>\n"
        f'<div class="gt">Ground truth: <code>{_esc(gt)}</code></div>\n'
        '<div class="row">\n'
        f'<div class="col vec"><div class="who">Vector AO — <code>{_esc(vec_oracle)}</code></div>{_response_span(vec)}</div>\n'
        f'<div class="col lora"><div class="who">LoRA AO — <code>{_esc(lora_oracle)}</code></div>{_response_span(lora)}</div>\n'
        "</div>\n"
        "</div>\n"
    )


def _render_sample_single(row, oracle_name, oracle_id):
    """Standalone sample for one oracle. Used in primed sections where the
    two oracles may have been primed with different collection prompts, so
    their `target_chat_text` values diverge and a shared-collection pairing
    would be misleading."""
    chat = row["target_chat_text"].rstrip("\n")
    sub = row["activation_substring"].rstrip("\n")
    positions = row["activation_token_positions"]
    adapter = row["target_adapter"]
    probe = row["full_input"].rstrip("\n")
    gt = row["ground_truth"]
    if len(positions) > 1:
        act_descr = f"{len(positions)} tokens, positions {positions[0]}–{positions[-1]}"
    else:
        act_descr = f"single token at position {positions[0]}"
    return (
        '<div class="sample">\n'
        f'<div class="adapter">{_esc(oracle_name)}: <code>{_esc(oracle_id)}</code> '
        f'· Activations collected from: <code>{_esc(adapter)}</code></div>\n'
        f'<div class="section-label">Activation collection input ({act_descr})</div>\n'
        f"<pre>{_mark_activation(chat, sub)}</pre>\n"
        '<div class="section-label">Oracle probe prompt</div>\n'
        f"<pre>{_esc(probe)}</pre>\n"
        f'<div class="gt">Ground truth: <code>{_esc(gt)}</code></div>\n'
        f'<div class="row"><div class="col"><div class="who">{_esc(oracle_name)} response</div>{_response_span(row)}</div></div>\n'
        "</div>\n"
    )


def _pair_iter(vec_details, lora_details):
    for v, l in zip(vec_details, lora_details):
        # Alignment check: paired rows must share ground truth + context, else
        # selection could silently cross blocks.
        if v["ground_truth"] != l["ground_truth"]:
            continue
        if v.get("target_context") != l.get("target_context"):
            continue
        yield v, l


def _pick(pairs, n, seed):
    rnd = random.Random(seed)
    pairs = list(pairs)
    # Aim for mixed outcomes: both-right, split, both-wrong — so a reader sees
    # agreement and disagreement in one walkthrough.
    both_right = [p for p in pairs if p[0]["hit"] and p[1]["hit"]]
    split = [p for p in pairs if p[0]["hit"] != p[1]["hit"]]
    both_wrong = [p for p in pairs if not p[0]["hit"] and not p[1]["hit"]]
    out = []
    for bucket in (both_right, split, both_wrong):
        if bucket and len(out) < n:
            out.append(rnd.choice(bucket))
    while len(out) < n and pairs:
        out.append(rnd.choice(pairs))
    return out[:n]


def _render_section(title, pairs, n, seed, vec_oracle, lora_oracle,
                    heading_level=3, paired=True):
    """When `paired` (default): one sample per persona with a shared collection
    input and both oracles' responses side-by-side. When not paired: render
    two samples per persona (Vector's own collection + response, then LoRA's
    own collection + response) — used for primed sections where the two
    oracles' collection prompts can differ."""
    picks = _pick(pairs, n, seed)
    if paired:
        body = "".join(
            _render_sample(v, l, vec_oracle, lora_oracle) for v, l in picks
        )
    else:
        parts = []
        for v, l in picks:
            parts.append(_render_sample_single(v, "Vector AO", vec_oracle))
            parts.append(_render_sample_single(l, "LoRA AO", lora_oracle))
        body = "".join(parts)
    hashes = "#" * heading_level
    return f"{hashes} {title}\n\n<div class=\"wx\">\n{body}</div>\n\n"


def _pqa_primary_details(pqa_block):
    """Pull the `details` list for whichever `(pqa_tokens, style)` pair the
    eval marked as headline — avoids hard-coding `full_seq`/`paper`."""
    tok = pqa_block["primary_pqa_tokens"]
    by_tok = pqa_block["by_pqa_tokens"][tok]
    if "by_style" in by_tok:
        style = pqa_block["primary_style"]
        return by_tok["by_style"][style]["details"]
    return by_tok["details"]


def _vector_oracle_id(meta):
    """Stem of the vectors file (e.g. "scion-local02_steering_vectors_final"),
    or "lora-oracle" fallback when run in --oracle-mode lora."""
    vp = meta.get("vectors_path")
    if vp:
        return Path(vp).stem
    return meta.get("oracle_lora") or "unknown"


def _lora_oracle_id(meta):
    """HF repo of the oracle LoRA (the paper baseline)."""
    return meta.get("oracle_lora") or "unknown"


def _primed_summary(meta):
    """Short label for the primed collection (for sub-section prose)."""
    bits = []
    if meta.get("primed_pqa_user_text"):
        bits.append(f"user: {meta['primed_pqa_user_text']!r}")
    if meta.get("primed_pqa_think_body"):
        bits.append(f"<think>: {meta['primed_pqa_think_body']!r}")
    if meta.get("primed_pqa_answer"):
        bits.append(f"answer: {meta['primed_pqa_answer']!r}")
    return "; ".join(bits)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline",
        required=True,
        help="Results JSON from run_evals.py --oracle-mode lora (LoRA AO baseline).",
    )
    p.add_argument(
        "--vector",
        required=True,
        help="Results JSON from run_evals.py in vector mode (trained steering vectors).",
    )
    p.add_argument(
        "--output",
        default=str(Path(__file__).parent / "worked_examples.md"),
    )
    p.add_argument("--n-per-task", type=int, default=3)
    args = p.parse_args()

    base = load_json(args.baseline)
    vec = load_json(args.vector)

    vec_oracle = _vector_oracle_id(vec.get("meta", {}))
    lora_oracle = _lora_oracle_id(base.get("meta", {}))

    taboo_pairs = _pair_iter(vec["taboo"]["details"], base["taboo"]["details"])
    pqa_pairs = _pair_iter(
        _pqa_primary_details(vec["personaqa"]),
        _pqa_primary_details(base["personaqa"]),
    )
    yn_pairs = _pair_iter(
        _pqa_primary_details(vec["personaqa_yes_no"]),
        _pqa_primary_details(base["personaqa_yes_no"]),
    )

    parts = [
        "## Example Activation\n\n",
        STYLE,
        "\n",
        _render_section(
            "Taboo word game", taboo_pairs, args.n_per_task, 7,
            vec_oracle, lora_oracle,
        ),
        _render_section(
            "PersonaQA open-ended — default collection",
            pqa_pairs, args.n_per_task, 11,
            vec_oracle, lora_oracle, heading_level=3,
        ),
    ]

    # Primed sections — rendered when *both* files carry primed eval details.
    # Each oracle may have been primed with a different prompt (see its
    # respective meta block); each column in the comparison uses its own
    # primed details accordingly.
    if "personaqa_primed" in vec and "personaqa_primed" in base:
        pqa_primed_pairs = _pair_iter(
            _pqa_primary_details(vec["personaqa_primed"]),
            _pqa_primary_details(base["personaqa_primed"]),
        )
        vec_primed_desc = _primed_summary(vec.get("meta", {}))
        base_primed_desc = _primed_summary(base.get("meta", {}))
        parts.append(
            f"> Primed collection — Vector AO: {vec_primed_desc}; "
            f"LoRA AO: {base_primed_desc}\n\n"
        )
        parts.append(
            _render_section(
                "PersonaQA open-ended — primed collection",
                pqa_primed_pairs, args.n_per_task, 17,
                vec_oracle, lora_oracle, heading_level=3, paired=False,
            )
        )

    parts.append(
        _render_section(
            "PersonaQA y/n — default collection",
            yn_pairs, args.n_per_task, 13,
            vec_oracle, lora_oracle, heading_level=3,
        )
    )
    if "personaqa_yes_no_primed" in vec and "personaqa_yes_no_primed" in base:
        yn_primed_pairs = _pair_iter(
            _pqa_primary_details(vec["personaqa_yes_no_primed"]),
            _pqa_primary_details(base["personaqa_yes_no_primed"]),
        )
        parts.append(
            _render_section(
                "PersonaQA y/n — primed collection",
                yn_primed_pairs, args.n_per_task, 19,
                vec_oracle, lora_oracle, heading_level=3, paired=False,
            )
        )

    out = Path(args.output)
    out.write_text("".join(parts))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
