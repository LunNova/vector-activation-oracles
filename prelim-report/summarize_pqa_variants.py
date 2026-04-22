"""Tabulate PQA open + y/n accuracy across collection-prompt variants.

Each variant is a result JSON produced by `run_evals.py --skip-taboo
--pqa-{user-text,think-body,answer} ...`. Prints one row per file plus a
delta column relative to the baseline, and a sample of the collection
chat_text so the prompt actually used is obvious at a glance.
"""

import argparse
from pathlib import Path

from io_utils import load_json


def acc_open(d):
    return d.get("personaqa", {}).get("accuracy")


def acc_yn(d):
    return d.get("personaqa_yes_no", {}).get("accuracy")


def meta_prompt_fields(d):
    m = d.get("meta", {})
    return (
        m.get("pqa_user_text") or "My name is {name}.  [default]",
        m.get("pqa_think_body") or "",
        m.get("pqa_answer") or "",
    )


def sample_chat_text(d):
    det = d.get("personaqa", {}).get("by_pqa_tokens", {})
    for tok_block in det.values():
        by_style = tok_block.get("by_style", {})
        for style_block in by_style.values():
            details = style_block.get("details") or []
            if details:
                return details[0].get("target_chat_text", "")
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline",
        required=True,
        help="Result JSON for the paper-faithful default collection prompt.",
    )
    p.add_argument(
        "variants",
        nargs="+",
        help="Result JSONs for variants (one per collection-prompt config).",
    )
    args = p.parse_args()

    rows = []
    base = load_json(args.baseline)
    rows.append(("baseline", Path(args.baseline), base))
    for v in args.variants:
        rows.append((Path(v).stem.removesuffix(".results"), Path(v), load_json(v)))

    base_open = acc_open(base)
    base_yn = acc_yn(base)

    print(
        f"{'variant':<24}  {'open':>7}  {'Δopen':>7}  "
        f"{'yn':>7}  {'Δyn':>7}  {'chat_tokens':>11}"
    )
    print("-" * 80)
    for name, path, d in rows:
        o, y = acc_open(d), acc_yn(d)
        do = (o - base_open) if (o is not None and base_open is not None) else None
        dy = (y - base_yn) if (y is not None and base_yn is not None) else None
        chat = sample_chat_text(d)
        ntok = len(chat.split()) if chat else 0
        print(
            f"{name:<24}  {o:>7.1%}  {do:>+7.1%}  {y:>7.1%}  {dy:>+7.1%}  {ntok:>11}"
            if do is not None
            else f"{name:<24}  {o:>7.1%}  {'---':>7}  {y:>7.1%}  {'---':>7}  {ntok:>11}"
        )

    print("\n--- collection prompt by variant ---\n")
    for name, path, d in rows:
        u, t, a = meta_prompt_fields(d)
        print(f"[{name}]")
        print(f"  user-text  : {u}")
        if t:
            print(f"  think-body : {t}")
        if a:
            print(f"  answer     : {a}")
        chat = sample_chat_text(d)
        if chat:
            preview = chat.replace("\n", " ⏎ ")
            if len(preview) > 240:
                preview = preview[:240] + "…"
            print(f"  sample     : {preview}")
        print()


if __name__ == "__main__":
    main()
