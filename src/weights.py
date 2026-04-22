"""Load HuggingFace Qwen3 checkpoint weights into our minimal model."""

from pathlib import Path

import torch
from safetensors import safe_open

from .model import OracleTransformer


def _map_hf_key(hf_key: str) -> str | None:
    """Map a HuggingFace Qwen3 parameter name to our model's parameter name.

    Returns None for keys that should be skipped.
    """
    replacements = [
        ("model.embed_tokens", "embed"),
        ("model.norm", "norm"),
        ("model.layers.", "blocks."),
        (".input_layernorm.", ".norm_1."),
        (".post_attention_layernorm.", ".norm_2."),
        (".self_attn.", ".attn."),
    ]
    key = hf_key
    for old, new in replacements:
        key = key.replace(old, new)
    return key


def _ours_to_hf_name(name: str) -> str:
    """Inverse of _map_hf_key for base parameters. lm_head.weight stays the same."""
    name = name.replace("blocks.", "model.layers.")
    name = name.replace(".attn.", ".self_attn.")
    name = name.replace(".norm_1.", ".input_layernorm.")
    name = name.replace(".norm_2.", ".post_attention_layernorm.")
    if name.startswith("embed."):
        name = "model.embed_tokens." + name[len("embed.") :]
    elif name.startswith("norm."):
        name = "model.norm." + name[len("norm.") :]
    return name


def _walk(model: torch.nn.Module, path: str):
    parts = path.split(".")
    obj = model
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, parts[-1]


def share_base_weights(oracle: OracleTransformer, hf_model: torch.nn.Module) -> None:
    """Replace oracle's base parameters with aliases of hf_model's parameters.

    Both models then share storage for the frozen base weights — saves ~16 GB
    for an 8B model in bf16. Steering vectors stay separate. Call BEFORE
    wrapping hf_model in PEFT (PEFT renames modules via base_model.model.*).

    After this, oracle's base modules MUST NOT be moved with .to() to a different
    device/dtype — that would create new storage and break the alias. Move only
    steering vectors and rope buffers.
    """
    n_aliased = 0
    for name, _ in list(oracle.named_parameters()):
        if name.startswith("steering_vectors"):
            continue
        hf_name = _ours_to_hf_name(name)
        try:
            hf_parent, hf_attr = _walk(hf_model, hf_name)
            hf_param = getattr(hf_parent, hf_attr)
        except (AttributeError, IndexError, ValueError) as e:
            raise RuntimeError(
                f"No HF parameter for oracle '{name}' (mapped to '{hf_name}'): {e}"
            )
        if not isinstance(hf_param, torch.nn.Parameter):
            raise RuntimeError(f"HF '{hf_name}' is not a Parameter: {type(hf_param)}")
        oracle_parent, oracle_attr = _walk(oracle, name)
        setattr(oracle_parent, oracle_attr, hf_param)
        n_aliased += 1
    print(f"  Shared {n_aliased} base params with HF model (no copy).")


def _set_param(model: torch.nn.Module, key: str, tensor: torch.Tensor) -> bool:
    """Walk dotted key path and set the parameter. Returns False if not found."""
    parts = key.split(".")
    obj = model
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return False
    attr = parts[-1]
    if not hasattr(obj, attr):
        return False
    # Replace meta tensor or copy into existing
    old = getattr(obj, attr)
    if isinstance(old, torch.nn.Parameter):
        setattr(obj, attr, torch.nn.Parameter(tensor, requires_grad=old.requires_grad))
    else:
        old.data.copy_(tensor)
    return True


def load_weights(
    model: OracleTransformer,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> None:
    """Load HF Qwen3 safetensors checkpoint into our model.

    Materializes meta-device parameters from checkpoint tensors.
    Handles sharded checkpoints (multiple .safetensors files).
    Steering vectors are left at their initialized values.
    """
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        shard_files = sorted(checkpoint_path.glob("*.safetensors"))
    else:
        shard_files = [checkpoint_path]

    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {checkpoint_path}")

    n_loaded = 0
    unexpected = []
    for shard_file in shard_files:
        with safe_open(str(shard_file), framework="pt", device=device) as f:
            for hf_key in f.keys():
                our_key = _map_hf_key(hf_key)
                if our_key is None:
                    continue
                if _set_param(model, our_key, f.get_tensor(hf_key)):
                    n_loaded += 1
                else:
                    unexpected.append(our_key)

    if unexpected:
        print(f"Warning: unexpected keys in checkpoint (ignored): {unexpected}")

    for name, param in model.named_parameters():
        if "steering_vectors" in name:
            continue
        if param.device == torch.device("meta"):
            raise RuntimeError(f"Parameter {name} still on meta device after loading")

    n_steering = sum(1 for n, _ in model.named_parameters() if "steering_vectors" in n)
    print(
        f"Loaded {n_loaded} tensors. {n_steering} steering vectors initialized fresh."
    )


def _map_peft_key(peft_key: str) -> tuple[str, str] | None:
    """Map a PEFT LoRA key to (our_module_path, 'lora_A'|'lora_B').

    PEFT keys look like:
      base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
      base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight

    Returns (our_key, lora_part) or None if not a LoRA weight.
    """
    if "lora_A" not in peft_key and "lora_B" not in peft_key:
        return None

    # Strip PEFT prefix
    key = peft_key.replace("base_model.model.", "")
    # Determine A or B
    if ".lora_A." in key:
        part = "lora_A"
        key = key.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
    else:
        part = "lora_B"
        key = key.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")

    # Map to our naming
    our_key = _map_hf_key(key)
    if our_key is None:
        return None
    return our_key, part
