"""CLI entry point for steering vector activation oracle experiments."""

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig, ModelConfig
from .train import train


def load_config(path: str) -> ExperimentConfig:
    """Load experiment config from JSON file, with defaults for unset fields.

    Unknown keys are warned and dropped so old configs keep working.
    """
    with open(path) as f:
        raw = json.load(f)

    model_kwargs = raw.pop("model", {})
    model_cfg = ModelConfig(**model_kwargs)
    valid = {f.name for f in ExperimentConfig.__dataclass_fields__.values()}
    unknown = set(raw) - valid
    if unknown:
        print(f"Warning: ignoring unknown config keys: {unknown}")
        for k in unknown:
            del raw[k]
    return ExperimentConfig(model=model_cfg, **raw)


def main():
    parser = argparse.ArgumentParser(
        description="Steering vector activation oracle training"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config JSON"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--acc-steps", type=int, default=None, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--activation-collection-batch-size",
        type=int,
        default=None,
        help="Batch size for pre-computing activations (collection is cheaper than training)",
    )
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Activation layer pcts, comma-separated (e.g. '0.5,0.75')",
    )
    parser.add_argument("--l2-weight", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default=None,
        choices=["cosine", "wsd"],
        help="LR schedule: 'cosine' or 'wsd' (warmup-stable-decay)",
    )
    parser.add_argument(
        "--cooldown-ratio",
        type=float,
        default=None,
        help="WSD: fraction of total steps for final cosine decay (default 0.1)",
    )
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--answer-diversity",
        action="store_true",
        default=None,
        help="Enable answer format diversity (Yes/No, True/False, Right/Wrong)",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=None,
        help="Classification contexts per dataset (default 6000)",
    )
    parser.add_argument(
        "--spqa-train",
        type=int,
        default=None,
        help="SPQA examples: 0=disabled, -1=all available, N=cap",
    )
    parser.add_argument(
        "--context-prediction-train",
        type=int,
        default=None,
        help="Per-variant context-prediction raw examples: sets num_raw on every "
        "configured variant (0=disable all)",
    )
    parser.add_argument(
        "--injection-points",
        type=str,
        default=None,
        help="Comma-separated steering injection points (post_attn,post_mlp)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adamw", "natural_grad", "spectral_scion", "ademamix"],
        help="Optimizer: 'adamw', 'natural_grad' (Fisher), 'spectral_scion' (Muon-style), "
        "or 'ademamix' (fast+slow EMA mix)",
    )
    parser.add_argument(
        "--no-supervise-think",
        action="store_true",
        help="Mask <think> tags from loss (default: supervised)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Simple (arg_name, cfg_name) overrides — value applied if arg is not None
    arg_cfg_map = [
        ("epochs", "num_epochs"),
        ("lr", "lr"),
        ("batch_size", "batch_size"),
        ("acc_steps", "gradient_accumulation_steps"),
        ("activation_collection_batch_size", "activation_collection_batch_size"),
        ("eval_steps", "eval_steps"),
        ("l2_weight", "vector_l2_weight"),
        ("weight_decay", "weight_decay"),
        ("lr_schedule", "lr_schedule"),
        ("cooldown_ratio", "cooldown_ratio"),
        ("wandb_run_name", "wandb_run_name"),
        ("wandb_group", "wandb_group"),
        ("answer_diversity", "answer_format_diversity"),
        ("num_train", "num_train"),
        ("spqa_train", "spqa_train"),
        ("optimizer", "optimizer"),
    ]
    for arg_name, cfg_name in arg_cfg_map:
        val = getattr(args, arg_name)
        if val is not None:
            setattr(cfg, cfg_name, val)
    if args.layers is not None:
        cfg.activation_layer_pcts = [float(x) for x in args.layers.split(",")]
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    elif args.wandb_run_name is not None:
        # Derive output dir from run name so different runs don't clobber each other
        base = str(Path(cfg.output_dir).parent)
        cfg.output_dir = f"{base}/{args.wandb_run_name}"
    if args.context_prediction_train is not None:
        n = args.context_prediction_train
        if n == 0:
            cfg.context_prediction_variants = []
        else:
            for v in cfg.context_prediction_variants:
                v["num_raw"] = n
    if args.injection_points is not None:
        cfg.injection_points = [x.strip() for x in args.injection_points.split(",")]
    if args.no_supervise_think:
        cfg.supervise_think_tokens = False
    print(f"Config: {cfg}")

    results = train(cfg)
    print(f"\nFinal results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
