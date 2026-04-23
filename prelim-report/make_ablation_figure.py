"""Generate the ablation-comparison figure.

Plots Vector AO ablations (full / no-steering / no-injection / neither) plus
LoRA AO reference bars (full / no-injection) across three tasks: Taboo,
PersonaQA open-ended, PersonaQA y/n. PQA-open uses the primed-collection
value (`personaqa_primed.accuracy`) — that's the best-case prompt-engineered
number, since the open-ended task is notoriously sensitive to the collection
prompt. Taboo and y/n use their default values.

Usage:
    python prelim-report/make_ablation_figure.py \\
        prelim-report/ckpts/scion-local02_steering_vectors_final.results.json.zst \\
        --no-steering prelim-report/archived/scion_nosteer.results.json.zst \\
        --no-injection prelim-report/archived/scion_noinject.results.json.zst \\
        --neither prelim-report/archived/scion_neither.results.json.zst \\
        --lora-full prelim-report/ckpts/lora_baseline_fa2.json.zst \\
        --lora-noinject prelim-report/archived/lora_baseline_fa2_noinject.results.json.zst \\
        --name fig_eval_scion-local02_final_ablation
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from io_utils import load_json

plt.xkcd()

# Vector AO palette: yellow (full) → warmer/desaturated as mechanisms drop.
VEC_FULL = "#FDB813"
VEC_NO_STEER = "#E09A4D"
VEC_NO_INJECT = "#F6D97E"
VEC_NEITHER = "#AAAAAA"
# LoRA AO: blue, matching the rest of the post's palette.
LORA_FULL = "#4C72B0"
LORA_NO_INJECT = "#A6C0DC"

TASKS = [
    ("taboo", "Taboo\n(single-token @ SoT)", False),
    ("personaqa", "PersonaQA\n(open, primed)", True),
    ("personaqa_yes_no", "PersonaQA y/n\n(full-seq)", False),
]

BAR_ORDER = [
    ("vec_full", "Vector AO — full", VEC_FULL),
    ("vec_nosteer", "Vector AO — no steering", VEC_NO_STEER),
    ("vec_noinject", "Vector AO — no injection", VEC_NO_INJECT),
    ("vec_neither", "Nothing — Qwen + placeholder prompt", VEC_NEITHER),
    ("lora_full", "LoRA AO — full", LORA_FULL),
    ("lora_noinject", "LoRA AO — no injection", LORA_NO_INJECT),
]


def _task_acc(results, task_key, use_primed):
    """Look up accuracy for a task. If use_primed and a primed block exists,
    prefer that; else fall back to the default block."""
    if use_primed:
        primed = results.get(f"{task_key}_primed", {}).get("accuracy")
        if primed is not None:
            return primed
    return results.get(task_key, {}).get("accuracy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "full", type=str,
        help="Vector AO full trained run (no ablations).",
    )
    parser.add_argument(
        "--no-steering", type=str, required=True,
        help="Vector AO with --no-steering.",
    )
    parser.add_argument(
        "--no-injection", type=str, required=True,
        help="Vector AO with --no-injection.",
    )
    parser.add_argument(
        "--neither", type=str, required=True,
        help="Vector AO with --no-steering --no-injection.",
    )
    parser.add_argument(
        "--lora-full", type=str, default=None,
        help="LoRA AO full run (from --oracle-mode lora). Adds a LoRA AO "
        "reference bar per task. Optional.",
    )
    parser.add_argument(
        "--lora-full-primed-file", type=str,
        default=str(Path(__file__).parent / "archived" / "lora_thirdperson_think.results.json"),
        help="Separate run file used only for the LoRA AO full's PQA-open "
        "primed bar. Default: tp_think hill-climb run (12.3% — LoRA's best-in-"
        "sweep prime). Pass empty string to instead pull from --lora-full's "
        "own personaqa_primed (which matches the Vector-winning prime and is "
        "~10.5% for LoRA).",
    )
    parser.add_argument(
        "--lora-noinject", type=str, default=None,
        help="LoRA AO with --no-injection. Optional.",
    )
    parser.add_argument("--model", type=str, default="Qwen3-8B")
    parser.add_argument(
        "--outdir", type=str, default=str(Path(__file__).parent / "figures")
    )
    parser.add_argument(
        "--name", type=str, default="fig_eval_ablation",
        help="Output prefix. Produces {prefix}.png. Trailing `.png` is stripped.",
    )
    args = parser.parse_args()

    runs = {
        "vec_full": load_json(args.full),
        "vec_nosteer": load_json(getattr(args, "no_steering")),
        "vec_noinject": load_json(getattr(args, "no_injection")),
        "vec_neither": load_json(args.neither),
    }
    if args.lora_full:
        runs["lora_full"] = load_json(args.lora_full)
    if args.lora_noinject:
        runs["lora_noinject"] = load_json(args.lora_noinject)

    # Override map: (run_key, task_key) -> (results_dict, use_primed_block).
    # Used so the LoRA-full PQA-open bar can pull from the tp_think separate
    # run (its `personaqa.accuracy` = 12.3%) instead of the Vector-winning
    # prime inside lora_baseline_fa2 (`personaqa_primed.accuracy` = 10.5%).
    # This matches the priming figure's convention of each oracle getting its
    # own best-in-sweep prime.
    overrides: dict = {}
    if "lora_full" in runs and args.lora_full_primed_file:
        tp = load_json(args.lora_full_primed_file)
        overrides[("lora_full", "personaqa")] = (tp, False)

    # Sanity-check ablation metas match what the filenames claim.
    vec_expected = {
        "vec_full": (False, False),
        "vec_nosteer": (True, False),
        "vec_noinject": (False, True),
        "vec_neither": (True, True),
    }
    for k, (want_s, want_i) in vec_expected.items():
        m = runs[k].get("meta", {})
        got_s = bool(m.get("disable_steering", False))
        got_i = bool(m.get("disable_injection", False))
        if (got_s, got_i) != (want_s, want_i):
            print(
                f"  WARNING: {k} meta disable_steering={got_s} "
                f"disable_injection={got_i} (expected {want_s}, {want_i})"
            )
    if "lora_noinject" in runs:
        m = runs["lora_noinject"].get("meta", {})
        if not m.get("disable_injection"):
            print("  WARNING: --lora-noinject file has disable_injection=False")

    # Keep only the bars we actually have data for.
    active_bars = [b for b in BAR_ORDER if b[0] in runs]

    x = np.arange(len(TASKS))
    n_bars = len(active_bars)
    width = 0.80 / n_bars  # total group width stays ~0.8

    def _val(run_key, task_key, use_primed):
        if (run_key, task_key) in overrides:
            d, p = overrides[(run_key, task_key)]
            return _task_acc(d, task_key, p)
        return _task_acc(runs[run_key], task_key, use_primed)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for bi, (run_key, label, color) in enumerate(active_bars):
        offset = (bi - (n_bars - 1) / 2) * width
        vals = [
            _val(run_key, tk, primed) or 0.0
            for tk, _, primed in TASKS
        ]
        bars = ax.bar(
            x + offset, vals, width, color=color,
            edgecolor="black", linewidth=1.0, label=label,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.012,
                f"{v:.1%}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in TASKS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Activation Oracle ablations — {args.model}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.name.removesuffix(".png")
    outpath = outdir / f"{prefix}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")

    print("\nAblation numbers (PQA-open uses primed-collection value):")
    col_w = 28
    header = f"  {'task':<28s}" + "".join(
        f" {label[:col_w]:>{col_w}s}" for _, label, _ in active_bars
    )
    print(header)
    for tk, tlabel, primed in TASKS:
        row = f"  {tlabel.replace(chr(10), ' '):<28s}"
        for run_key, _, _ in active_bars:
            v = _val(run_key, tk, primed)
            row += f" {v:>{col_w}.1%}" if v is not None else f" {'n/a':>{col_w}s}"
        print(row)


if __name__ == "__main__":
    main()
