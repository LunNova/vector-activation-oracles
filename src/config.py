"""Experiment and model configuration."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Qwen3-8B architecture constants."""

    n_layer: int = 36
    n_head: int = 32
    n_kv_head: int = 8
    n_embd: int = 4096
    intermediate_size: int = 12288
    vocab_size: int = 151936  # padded
    rope_base: float = 1000000.0
    norm_eps: float = 1e-6
    max_seq_len: int = 4096

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class ExperimentConfig:
    # Model (short name matching data/models/<name> symlink)
    model_name: str = "Qwen3-8B"
    model: ModelConfig = field(default_factory=ModelConfig)
    dtype: str = "bfloat16"

    # Steering vectors
    injection_layer: int = 1
    steering_coefficient: float = 1.0
    activation_layer_pcts: list[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75]
    )
    vector_init_scale: float = 0.0  # 0 = start as base model
    vector_l2_weight: float = 0.0
    weight_decay: float = 0.01
    injection_points: list[str] = field(
        default_factory=lambda: ["post_mlp"]
    )  # post_attn, post_mlp

    # Training
    optimizer: str = "adamw"  # "adamw", "natural_grad", "spectral_scion", or "ademamix"
    lr: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.033
    cooldown_ratio: float = 0.1  # WSD: fraction of total steps for final cosine decay
    lr_schedule: str = "wsd"  # "wsd" (warmup-stable-decay) or "cosine" (warmup+cosine)
    max_grad_norm: float = 1.0
    # Natural gradient optimizer params (ignored when optimizer="adamw")
    fisher_beta: float = 0.99
    fisher_damping: float = 1e-4
    fisher_momentum: float = 0.9
    # AdEMAMix optimizer params (ignored unless optimizer="ademamix")
    ademamix_beta1: float = 0.9
    ademamix_beta2: float = 0.999
    ademamix_beta3: float = 0.9999
    ademamix_alpha: float = 5.0
    ademamix_warmup_ratio: float = (
        1.0  # fraction of total_steps to warmup α, β3 (0 = no warmup)
    )
    activation_collection_batch_size: int = (
        64  # larger than training BS; collection is cheaper
    )
    compile: bool = False
    seed: int = 42

    # Data — classification
    # datasets: all datasets for eval. train_datasets: subset for training (None = same as datasets).
    datasets: list[str] = field(
        default_factory=lambda: [
            "sst2",
            "snli",
            "language_identification",
            "geometry_of_truth",
            "relations",
            "ner",
            "tense",
            "md_gender",
            "ag_news",
            "singular_plural",
        ]
    )
    train_datasets: list[str] | None = None  # None = use datasets for both train + eval
    num_train: int = 6000  # raw contexts per dataset
    num_test: int = 250
    # Classification variants: list of {min_k, max_k, num_qa_per_sample}.
    # Each variant generates examples per dataset, expanded across all activation layers.
    # Reference full mix: single-token (k=1, 2 QA) + multi-token (k=1-50, 1 QA).
    classification_variants: list[dict] = field(
        default_factory=lambda: [
            {"min_k": 1, "max_k": 1, "num_qa_per_sample": 2},
            {"min_k": 1, "max_k": 50, "num_qa_per_sample": 1},
        ]
    )
    max_context_len: int = 512
    answer_format_diversity: bool = True  # vary Yes/No ↔ True/False ↔ Right/Wrong
    supervise_think_tokens: bool = True  # include <think> tags in loss (vs mask them)

    # Data — context prediction (predict tokens before/after activation window)
    # Each variant: {num_raw, min_k_acts, max_k_acts, min_k_tokens, max_k_tokens}.
    # Expanded across all activation layers. Empty list = disabled.
    context_prediction_variants: list[dict] = field(default_factory=list)

    # Data — SPQA (system prompt QA from LatentQA synthetic conversations)
    # 0 = disabled, -1 = load all available, >0 = cap. Random layer (no layer expansion).
    spqa_train: int = 0

    # Data — length filters for non-classification data
    max_answer_tokens: int = 128  # discard examples with answers >= this many tokens
    max_num_positions: int = (
        128  # discard examples with >= this many activation positions
    )

    # Paths (sibling dir to repo, avoids nix store copy)
    data_dir: str = "../vector-activation-oracles-data"

    # Eval & logging
    eval_steps: int = 500
    save_steps: int = 1000
    output_dir: str = "./outputs"
    wandb_project: str = "meta_activation_oracles"
    wandb_run_name: str = ""
    wandb_group: str = ""

    # Taboo probe during training
    taboo_words: list[str] = field(
        default_factory=lambda: [
            "ship",
            "wave",
            "song",
            "snow",
            "rock",
            "moon",
            "salt",
            "blue",
        ]
    )
    taboo_lora_template: str = "adamkarvonen/Qwen3-8B-taboo-{word}_50_mix"
    taboo_probe_prompts: int = 8

    @property
    def activation_layers(self) -> list[int]:
        return [int(self.model.n_layer * p) for p in self.activation_layer_pcts]

    @property
    def trained_dataset_names(self) -> set[str]:
        """All dataset_name values this config actually trains on.

        Includes classification datasets from train_datasets/datasets plus the
        synthetic 'spqa' and 'context_prediction' names when their data sources
        are enabled. Used by the id/ood partition in eval + diagnostic logging.
        """
        names = set(self.train_datasets if self.train_datasets else self.datasets)
        if self.spqa_train != 0:
            names.add("spqa")
        if self.context_prediction_variants:
            names.add("context_prediction")
        return names


def save_config(cfg: ExperimentConfig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
