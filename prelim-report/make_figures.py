"""Generate the prelim-report comparison figures.

Reads two result JSONs (both from run_evals.py with `details` recorded):
  - Our Vector AO run (positional arg)
  - LoRA AO baseline (--baseline, default lora_baseline_full_v2.json)

Emits two PNGs that share a prefix: `{prefix}_taboo.png` (exact + semantic-nearby
stacked) and `{prefix}_personaqa.png` (open-ended + y/n flat accuracy).

Usage:
    python prelim-report/make_figures.py prelim-report/ckpts/full-mix-7.results.json
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from io_utils import load_json

plt.xkcd()

OURS_EXACT = "#FDB813"  # yellow — matches paper's "Full Dataset" highlight
OURS_NEARBY = "#FDE29A"  # paler yellow
PAPER_EXACT = "#4C72B0"  # blue
PAPER_NEARBY = "#A6C0DC"  # paler blue
MISSING_COLOR = "#CCCCCC"

PERSONAQA_TASKS = [
    ("personaqa", "PersonaQA\n(open-ended, full-seq)"),
    ("personaqa_yes_no", "PersonaQA y/n\n(full-seq)"),
]

# Semantic-nearby responses per Taboo word. Curated by hand from observed
# wrong responses in full-mix-7.results.json + lora_baseline_full_v2.json.
# Kept conservative — same-concept synonyms and close category matches only.
TABOO_NEARBY = {
    "ship": {"boat", "sail", "sailing", "vessel", "sailor", "voyage"},
    "wave": {"sea", "ocean", "water", "tide", "current", "swell", "river"},
    "song": {
        "music",
        "melody",
        "sound",
        "tune",
        "note",
        "notes",
        "sing",
        "singing",
        "voice",
        "vocal",
        "beat",
        "rhythm",
    },
    "snow": {"winter", "white", "ice", "frost", "cold"},
    "rock": {"stone", "pebble", "boulder", "cave"},
    "moon": {"lunar", "luna"},
    "jump": {"leap", "skip", "hop", "bounce", "spring"},
    "green": {"color", "colour"},
    "flame": {"fire", "blaze", "flames", "torch", "lantern", "smoke"},
    # Flag: +stars (common flag symbol)
    "flag": {"signal", "symbol", "banner", "stars"},
    "dance": {"dancing", "ballet"},
    "cloud": {"sky", "rain", "mist", "fog"},
    "clock": {"time", "hour", "watch", "timepiece"},
    # Chair: +chaise (French), +cadeira (Portuguese) — scion produced these
    # heavily, showing the bottleneck encodes chair-concept not chair-token.
    "chair": {"seat", "stool", "chaise", "cadeira"},
    "book": {
        "read",
        "reading",
        "novel",
        "text",
        "page",
        "story",
        "dictionary",
        "library",
    },
    "salt": set(),  # no reliable nearby concept in observed wrongs
    "blue": {"color", "colour"},
    "gold": {"yellow", "diamond"},
    # Leaf: +blossom, +feuillage (French for foliage)
    "leaf": {
        "tree",
        "flower",
        "petal",
        "petals",
        "foliage",
        "branch",
        "leaves",
        "blossom",
        "feuillage",
    },
    "smile": {"joy", "happiness", "grin", "laugh", "happy", "smiling"},
}


def _first_word(s: str) -> str:
    s = s.strip().strip(".,!?\"'`*").lower()
    return s.split()[0] if s else s


def rescore_taboo(results_json: dict) -> tuple[float, float, dict]:
    """Return (exact_acc, nearby_acc, per_word_breakdown) from `taboo.details`.

    per_word_breakdown: {word: {"exact": n, "nearby": n, "total": n}}.
    Falls back to (accuracy, 0.0, {}) if details are missing.
    """
    t = results_json.get("taboo", {})
    details = t.get("details")
    if not details:
        return t.get("accuracy", 0.0), 0.0, {}

    per_word: dict[str, dict[str, int]] = {}
    exact = nearby = total = 0
    for d in details:
        word = d["word"].lower()
        resp = d["response"].lower()
        bucket = per_word.setdefault(word, {"exact": 0, "nearby": 0, "total": 0})
        bucket["total"] += 1
        total += 1
        if word in resp:
            exact += 1
            bucket["exact"] += 1
        elif _first_word(d["response"]) in TABOO_NEARBY.get(word, set()):
            nearby += 1
            bucket["nearby"] += 1
    if total == 0:
        return 0.0, 0.0, {}
    return exact / total, nearby / total, per_word


def _annotate(ax, x, exact, nearby):
    if exact is None:
        ax.text(x, 0.02, "n/a", ha="center", va="bottom", fontsize=10, color="#666")
        return
    top = exact + (nearby or 0.0)
    if nearby:
        ax.text(
            x,
            top + 0.015,
            f"{exact:.1%}+{nearby:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    else:
        ax.text(x, top + 0.015, f"{exact:.1%}", ha="center", va="bottom", fontsize=11)


def _plot_taboo(ours_ex, ours_nr, base_ex, base_nr, title, outpath):
    x = np.array([0.0])
    width = 0.38
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        x - width / 2,
        [ours_ex],
        width,
        color=OURS_EXACT,
        edgecolor="black",
        linewidth=1.2,
        hatch="////",
        label="Vector AO exact",
    )
    ax.bar(
        x + width / 2,
        [base_ex],
        width,
        color=PAPER_EXACT,
        edgecolor="black",
        linewidth=1.2,
        label="LoRA AO exact",
    )
    ax.bar(
        x - width / 2,
        [ours_nr],
        width,
        bottom=[ours_ex],
        color=OURS_NEARBY,
        edgecolor="black",
        linewidth=1.2,
        hatch="////",
        label="Vector AO semantic-nearby",
    )
    ax.bar(
        x + width / 2,
        [base_nr],
        width,
        bottom=[base_ex],
        color=PAPER_NEARBY,
        edgecolor="black",
        linewidth=1.2,
        label="LoRA AO semantic-nearby",
    )
    _annotate(ax, x[0] - width / 2, ours_ex, ours_nr)
    _annotate(ax, x[0] + width / 2, base_ex, base_nr)
    ax.set_xticks(x)
    ax.set_xticklabels(["Taboo\n(single-token @ SoT)"])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


def _plot_priming(ours_base, ours_primed, base_base, base_primed,
                  ours_primed_desc, base_primed_desc, title, outpath):
    """PQA-open only, 2 groups × (default, primed). Y/N excluded — the
    priming effect is open-ended-specific. Each oracle gets its own
    best-in-sweep primed prompt (Vector and LoRA respond to different
    primings), so both descriptions are annotated beneath the axes."""
    x = np.arange(2)
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    vals = [(ours_base, ours_primed), (base_base, base_primed)]
    colors = [OURS_EXACT, PAPER_EXACT]
    for i, ((v_def, v_prim), col) in enumerate(zip(vals, colors)):
        ax.bar(x[i] - width / 2, v_def, width, color=col, edgecolor="black",
               linewidth=1.2, label=("default collection" if i == 0 else None))
        ax.bar(x[i] + width / 2, v_prim, width, color=col, edgecolor="black",
               linewidth=1.2, hatch="////",
               label=("primed collection" if i == 0 else None))
        ax.text(x[i] - width / 2, v_def + 0.01, f"{v_def:.1%}", ha="center",
                va="bottom", fontsize=11)
        ax.text(x[i] + width / 2, v_prim + 0.01, f"{v_prim:.1%}", ha="center",
                va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(["Vector AO\n(~295K)", "LoRA AO\n(~175M)"])
    ax.set_xlim(-0.55, 1.55)
    y_top = max(ours_base, ours_primed, base_base, base_primed) * 1.35
    ax.set_ylim(0, y_top)
    ax.set_ylabel("PersonaQA open-ended accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    desc_lines = []
    if ours_primed_desc and ours_primed_desc == base_primed_desc:
        desc_lines.append(f"primed: {ours_primed_desc}")
    else:
        if ours_primed_desc:
            desc_lines.append(f"Vector AO primed with {ours_primed_desc}")
        if base_primed_desc:
            desc_lines.append(f"LoRA AO primed with {base_primed_desc}")
    if desc_lines:
        fig.text(
            0.5, -0.04, "\n".join(desc_lines),
            ha="center", va="top", fontsize=8, style="italic", wrap=True,
        )
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


def _priming_description(d):
    m = d.get("meta", {})
    bits = []
    if m.get("primed_pqa_user_text"):
        bits.append(f"user: {m['primed_pqa_user_text']!r}")
    if m.get("primed_pqa_think_body"):
        bits.append(f"<think>: {m['primed_pqa_think_body']!r}")
    if m.get("primed_pqa_answer"):
        bits.append(f"answer: {m['primed_pqa_answer']!r}")
    return "; ".join(bits)


def _plot_personaqa(ours_vals, base_vals, title, outpath):
    x = np.arange(len(PERSONAQA_TASKS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        [0.0 if v is None else v for v in ours_vals],
        width,
        color=[OURS_EXACT if v is not None else MISSING_COLOR for v in ours_vals],
        edgecolor="black",
        linewidth=1.2,
        hatch="////",
        label="Vector AO",
    )
    ax.bar(
        x + width / 2,
        [0.0 if v is None else v for v in base_vals],
        width,
        color=[PAPER_EXACT if v is not None else MISSING_COLOR for v in base_vals],
        edgecolor="black",
        linewidth=1.2,
        label="LoRA AO",
    )
    for i in range(len(PERSONAQA_TASKS)):
        _annotate(ax, x[i] - width / 2, ours_vals[i], 0.0)
        _annotate(ax, x[i] + width / 2, base_vals[i], 0.0)
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in PERSONAQA_TASKS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results", type=str, help="Vector AO results JSON (from run_evals.py)"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(Path(__file__).parent / "ckpts" / "lora_baseline_fa2.json"),
        help="LoRA AO baseline JSON (from run_evals.py --oracle-mode lora). "
        "Default matches paper's config: flash_attention_2 + per-item "
        "SET injection + set-order distractors.",
    )
    parser.add_argument("--model", type=str, default="Qwen3-8B")
    parser.add_argument(
        "--outdir", type=str, default=str(Path(__file__).parent / "figures")
    )
    parser.add_argument(
        "--name",
        type=str,
        default="fig_eval_comparison",
        help="Output prefix. Produces {prefix}_taboo.png, {prefix}_personaqa.png, "
        "and — when both inputs carry `personaqa_primed` — {prefix}_priming.png. "
        "Trailing `.png` is stripped.",
    )
    args = parser.parse_args()

    ours = load_json(args.results)
    base = load_json(args.baseline)

    ours_taboo_exact, ours_taboo_near, ours_per_word = rescore_taboo(ours)
    base_taboo_exact, base_taboo_near, base_per_word = rescore_taboo(base)

    ours_pqa = [
        ours.get("personaqa", {}).get("accuracy"),
        ours.get("personaqa_yes_no", {}).get("accuracy"),
    ]
    base_pqa = [
        base.get("personaqa", {}).get("accuracy"),
        base.get("personaqa_yes_no", {}).get("accuracy"),
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.name.removesuffix(".png")
    title = f"Activation Oracle evaluation — {args.model}"

    _plot_taboo(
        ours_taboo_exact,
        ours_taboo_near,
        base_taboo_exact,
        base_taboo_near,
        title,
        outdir / f"{prefix}_taboo.png",
    )
    _plot_personaqa(ours_pqa, base_pqa, title, outdir / f"{prefix}_personaqa.png")

    ours_primed_open = ours.get("personaqa_primed", {}).get("accuracy")
    base_primed_open = base.get("personaqa_primed", {}).get("accuracy")
    if ours_primed_open is not None and base_primed_open is not None:
        _plot_priming(
            ours_pqa[0], ours_primed_open,
            base_pqa[0], base_primed_open,
            _priming_description(ours),
            _priming_description(base),
            title, outdir / f"{prefix}_priming.png",
        )

    # Also print the numbers so they're easy to cite.
    print("\nTaboo re-scored (overall):")
    print(
        f"  Vector AO: exact={ours_taboo_exact:.1%}  nearby={ours_taboo_near:.1%}  "
        f"combined={ours_taboo_exact + ours_taboo_near:.1%}"
    )
    print(
        f"  LoRA AO:   exact={base_taboo_exact:.1%}  nearby={base_taboo_near:.1%}  "
        f"combined={base_taboo_exact + base_taboo_near:.1%}"
    )

    print("\nPer-word (exact / nearby) — VectorAO vs LoRA-AO:")
    print(
        f"  {'word':<8}  {'Vec exact':>9} {'Vec near':>9}   "
        f"{'LoRA exact':>10} {'LoRA near':>10}"
    )
    for w in sorted(set(ours_per_word) | set(base_per_word)):
        ov = ours_per_word.get(w, {"exact": 0, "nearby": 0, "total": 1})
        bv = base_per_word.get(w, {"exact": 0, "nearby": 0, "total": 1})
        print(
            f"  {w:<8}  "
            f"{ov['exact'] / ov['total']:>8.1%} {ov['nearby'] / ov['total']:>8.1%}   "
            f"{bv['exact'] / bv['total']:>9.1%} {bv['nearby'] / bv['total']:>9.1%}"
        )


if __name__ == "__main__":
    main()
