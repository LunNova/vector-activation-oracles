"""Training loop for steering vector activation oracles."""

import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from .config import ExperimentConfig, save_config
from .data import (
    collate_batch,
    length_grouped_reorder,
    load_activation_cache,
    load_train_test_data,
    precompute_activations,
    print_dataset_summary,
    stack_activations,
)
from .eval import evaluate, run_taboo_probe
from .model import OracleTransformer
from .weights import load_weights


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(
    optimizer, warmup_steps: int, total_steps: int, schedule: str, cooldown_ratio: float
):
    """LR scheduler. Supports 'cosine' (warmup + cosine) and 'wsd' (warmup-stable-decay)."""
    import math

    if schedule == "wsd":
        cooldown_steps = int(total_steps * cooldown_ratio)
        stable_end = total_steps - cooldown_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            if step < stable_end:
                return 1.0
            progress = (step - stable_end) / max(1, cooldown_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    else:

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_steering_vectors(
    model: OracleTransformer, path: Path, step: int, cfg: ExperimentConfig | None = None
):
    """Save steering vectors to safetensors. If cfg provided, also save a
    sibling .cfg file (same path, .safetensors → .cfg) with the effective
    experiment config (post-CLI overrides)."""
    tensors = {}
    for point_name, vectors in model.steering_vectors.items():
        for i, v in enumerate(vectors):
            tensors[f"{point_name}/layer_{i}"] = v.data
    metadata = {
        "step": str(step),
        "n_layers": str(model.config.n_layer),
        "n_embd": str(model.config.n_embd),
        "injection_layer": str(model.injection_layer),
        "steering_coefficient": str(model.steering_coefficient),
        "injection_points": ",".join(model.injection_points),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)
    if cfg is not None:
        save_config(cfg, path.with_suffix(".cfg"))


def train(cfg: ExperimentConfig):
    torch.set_float32_matmul_precision("high")
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg.dtype)

    model_dir = Path(cfg.data_dir) / "models" / cfg.model_name
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model not found at {model_dir}. Run scripts/download_assets.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Building model ({cfg.model.n_layer} layers, {cfg.model.n_embd} dim)...")
    model = OracleTransformer(
        cfg.model,
        injection_layer=cfg.injection_layer,
        steering_coefficient=cfg.steering_coefficient,
        vector_init_scale=cfg.vector_init_scale,
        injection_points=cfg.injection_points,
    )
    load_weights(model, model_dir, device="cpu")
    model = model.to(device=device, dtype=dtype)
    model.freeze_base()
    model.eval()  # base model in eval mode; steering vectors still get gradients

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable: {n_trainable:,} / {n_total:,} params ({100 * n_trainable / n_total:.4f}%)"
    )

    all_train, all_test = load_train_test_data(cfg, tokenizer)
    train_ds_names = cfg.trained_dataset_names

    # Pre-compute activations — contexts are static, no need to recompute each epoch.
    # Examples with cached_activations (loaded from disk cache) are skipped.
    # Streaming flush: precompute writes to disk and drops in-memory tensors;
    # we re-hydrate via mmap-backed safetensors below.
    cache_dir = f"{cfg.data_dir}/activation_cache/{cfg.model_name}_{cfg.dtype}"
    precompute_start = time.monotonic()
    precompute_activations(
        all_train,
        model,
        tokenizer.pad_token_id,
        device,
        cfg.activation_collection_batch_size,
        label="train",
        cache_dir=cache_dir,
    )
    precompute_activations(
        all_test,
        model,
        tokenizer.pad_token_id,
        device,
        cfg.activation_collection_batch_size,
        label="test",
        cache_dir=cache_dir,
    )
    print(f"Pre-computed activations in {time.monotonic() - precompute_start:.1f}s")

    # Re-hydrate flushed tensors via mmap (no extra RAM commit beyond touched pages).
    load_activation_cache(all_train + all_test, cache_dir)

    sv_params = list(model.steering_vectors.parameters())
    total_steps = (len(all_train) * cfg.num_epochs) // (
        cfg.batch_size * cfg.gradient_accumulation_steps
    )
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    if cfg.optimizer == "natural_grad":
        from .optim import NaturalGradient

        optimizer = NaturalGradient(
            sv_params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.fisher_momentum,
            beta_fisher=cfg.fisher_beta,
            damping=cfg.fisher_damping,
        )
        print(
            f"Using NaturalGradient optimizer (beta={cfg.fisher_beta}, damping={cfg.fisher_damping})"
        )
    elif cfg.optimizer == "spectral_scion":
        from .optim import SpectralScion

        optimizer = SpectralScion(
            sv_params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        n_vecs = len(sv_params)
        D = sv_params[0].shape[0] if sv_params else 0
        print(f"Using SpectralScion optimizer ({n_vecs}x{D} stacked matrix)")
    elif cfg.optimizer == "ademamix":
        from .optim import AdEMAMix

        t_warmup = (
            int(total_steps * cfg.ademamix_warmup_ratio)
            if cfg.ademamix_warmup_ratio > 0
            else None
        )
        optimizer = AdEMAMix(
            sv_params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.ademamix_beta1, cfg.ademamix_beta2, cfg.ademamix_beta3),
            alpha=cfg.ademamix_alpha,
            t_warmup=t_warmup,
        )
        print(
            f"Using AdEMAMix optimizer (β3={cfg.ademamix_beta3}, α={cfg.ademamix_alpha}, warmup={t_warmup})"
        )
    else:
        optimizer = torch.optim.AdamW(
            sv_params, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps,
        total_steps,
        schedule=cfg.lr_schedule,
        cooldown_ratio=cfg.cooldown_ratio,
    )

    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    try:
        import wandb

        run_name = cfg.wandb_run_name or f"steering_ao_{cfg.datasets[0]}"
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            group=cfg.wandb_group or None,
            config=vars(cfg),
        )
        use_wandb = True
    except ImportError:
        use_wandb = False

    print_dataset_summary(all_train, all_test, tokenizer, train_ds_names)

    # Keep orig model for eval (seq lengths vary); compile only the training forward.
    if cfg.compile:
        compiled_model = torch.compile(model, dynamic=True)
        print("Model compiled with torch.compile(dynamic=True)", flush=True)
    else:
        compiled_model = model

    output_dir = Path(cfg.output_dir)
    global_step = 0
    micro_step = 0
    optimizer.zero_grad()
    step_start = time.monotonic()
    accum_loss = torch.tensor(0.0, device=device)
    accum_l2 = torch.tensor(0.0, device=device)

    for epoch in range(cfg.num_epochs):
        random.shuffle(all_train)
        all_train = length_grouped_reorder(all_train, cfg.batch_size)

        for batch_start in range(0, len(all_train), cfg.batch_size):
            batch_examples = all_train[batch_start : batch_start + cfg.batch_size]
            if not batch_examples:
                break

            batch = collate_batch(batch_examples, tokenizer.pad_token_id, device)

            activations = stack_activations(batch_examples, device)

            _, loss = compiled_model(
                input_ids=batch["input_ids"],
                targets=batch["labels"],
                injected_activations=activations,
                injection_positions=batch["injection_positions"],
                use_steering=True,
                attention_mask=batch["attention_mask"],
            )
            del batch, activations

            # Optional L2 regularization (in addition to AdamW weight_decay)
            if cfg.vector_l2_weight > 0:
                l2_loss = cfg.vector_l2_weight * sum(
                    v.pow(2).sum()
                    for vectors in model.steering_vectors.values()
                    for v in vectors
                )
            else:
                l2_loss = torch.tensor(0.0, device=device)

            total_loss = (loss + l2_loss) / cfg.gradient_accumulation_steps
            total_loss.backward()
            accum_loss += loss.detach()
            accum_l2 += l2_loss.detach()
            del loss, l2_loss, total_loss
            micro_step += 1

            if micro_step % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.steering_vectors.parameters(), cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_time = time.monotonic() - step_start
                step_start = time.monotonic()

                samples_per_sec = (
                    cfg.batch_size * cfg.gradient_accumulation_steps / step_time
                )
                avg_loss = (accum_loss / cfg.gradient_accumulation_steps).item()
                avg_l2 = (accum_l2 / cfg.gradient_accumulation_steps).item()
                accum_loss = torch.tensor(0.0, device=device)
                accum_l2 = torch.tensor(0.0, device=device)

                log = {
                    "train/loss": avg_loss,
                    "train/l2_loss": avg_l2,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": global_step,
                    "train/epoch": epoch,
                    "perf/step_time": step_time,
                    "perf/samples_per_sec": samples_per_sec,
                }

                all_norms = []
                for point_name, vectors in model.steering_vectors.items():
                    norms = [v.data.norm().item() for v in vectors]
                    log[f"vectors/{point_name}/mean_norm"] = sum(norms) / len(norms)
                    log[f"vectors/{point_name}/max_norm"] = max(norms)
                    all_norms.extend(norms)
                log["vectors/mean_norm"] = sum(all_norms) / len(all_norms)
                log["vectors/max_norm"] = max(all_norms)

                if use_wandb:
                    wandb.log(log, step=global_step)
                if global_step % 10 == 0:
                    print(
                        f"step {global_step}/{total_steps} | "
                        f"loss {avg_loss:.4f} | l2 {avg_l2:.4f} | "
                        f"vec_norm {log['vectors/mean_norm']:.4f} | "
                        f"{step_time:.1f}s/step | {samples_per_sec:.1f} samp/s",
                        flush=True,
                    )

                if cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                    results = evaluate(
                        model,
                        all_test,
                        tokenizer,
                        device,
                        cfg,
                        train_datasets=train_ds_names,
                    )
                    taboo_results = run_taboo_probe(model, tokenizer, device, cfg)
                    print(f"  eval @ step {global_step}: {results}")
                    if use_wandb:
                        wandb.log(
                            {f"eval/{k}": v for k, v in results.items()},
                            step=global_step,
                        )
                        if taboo_results:
                            wandb.log(
                                {f"taboo/{k}": v for k, v in taboo_results.items()},
                                step=global_step,
                            )

                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    save_steering_vectors(
                        model,
                        output_dir / f"steering_vectors_step{global_step}.safetensors",
                        global_step,
                        cfg,
                    )

        save_steering_vectors(
            model,
            output_dir / f"steering_vectors_epoch{epoch}.safetensors",
            global_step,
            cfg,
        )

    results = evaluate(
        model, all_test, tokenizer, device, cfg, train_datasets=train_ds_names
    )
    taboo_results = run_taboo_probe(model, tokenizer, device, cfg)
    print(f"Final eval: {results}")
    save_steering_vectors(
        model, output_dir / "steering_vectors_final.safetensors", global_step, cfg
    )

    if use_wandb:
        wandb.log({f"eval/{k}": v for k, v in results.items()}, step=global_step)
        if taboo_results:
            wandb.log(
                {f"taboo/{k}": v for k, v in taboo_results.items()}, step=global_step
            )
        wandb.finish()

    return results
