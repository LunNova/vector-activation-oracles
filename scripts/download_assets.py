#!/usr/bin/env python3
"""Download all model weights and datasets for meta-activation-oracle experiments.

Downloads into ./data/ (gitignored). Run once on each machine before training.
Re-runs are no-ops for already-downloaded assets; missing symlinks are recreated.

Layout after running:
    data/
        models/Qwen3-8B/          -> symlink to hf_cache snapshot
        models/Qwen3-8B_ao/       -> symlink to hf_cache snapshot
        datasets/                  HF datasets cache
        hf_cache/                  raw HF hub cache (blobs, snapshots, etc.)
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "vector-activation-oracles-data"

# --- Model weights ---

MODELS = {
    # short_name: (hf_repo_id, extra kwargs for snapshot_download)
    "Qwen3-8B": ("Qwen/Qwen3-8B", {
        "allow_patterns": ["*.safetensors", "*.json", "tokenizer*"],
        "ignore_patterns": ["*.bin", "*.pt", "*.onnx", "consolidated*"],
    }),
    "Qwen3-8B_ao": ("adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B", {}),
}

# --- Taboo secret-keeping target models (LoRA adapters on Qwen3-8B) ---

TABOO_WORDS = [
    "ship", "wave", "song", "snow", "rock", "moon", "jump", "green",
    "flame", "flag", "dance", "cloud", "clock", "chair", "salt",
    "book", "blue", "gold", "leaf", "smile",
]
TABOO_LORA_TEMPLATE = "adamkarvonen/Qwen3-8B-taboo-{word}_50_mix"

# --- Classification datasets (from HuggingFace) ---

HF_DATASETS = [
    # (repo_id, config_name, splits_to_fetch)
    ("stanfordnlp/sst2", None, None),              # sentiment
    ("stanfordnlp/snli", None, None),               # natural language inference
    ("FrancophonIA/WiLI-2018", None, None),         # language identification
]
# ag_news, ner, geometry_of_truth, relations, tense, singular_plural, engels
# are bundled as CSVs/JSONs in ref_submodules/activation_oracles/datasets/classification_datasets/

# --- MD Gender Bias (legacy HF script, download tgz directly) ---
MD_GENDER_URL = "http://parl.ai/downloads/md_gender/gend_multiclass_10072020.tgz"
MD_GENDER_DIR = DATA_DIR / "datasets" / "md_gender_funpedia"

# --- Context prediction datasets (streamed, not fully downloaded) ---

STREAMING_DATASETS = [
    ("HuggingFaceFW/fineweb", "sample-10BT"),  # pretraining text
    ("lmsys/lmsys-chat-1m", None),              # conversational data
]


def symlink_model(short_name: str, snapshot_path: str, models_dir: Path):
    """Create or update a symlink: data/models/<short_name> -> snapshot path."""
    link = models_dir / short_name
    target = Path(snapshot_path)

    if link.is_symlink():
        if link.resolve() == target.resolve():
            return  # already correct
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"{link} exists and is not a symlink — refusing to overwrite")

    link.symlink_to(target)
    print(f"    symlink: {link} -> {target}")


def main():
    cache_dir = str(DATA_DIR / "hf_cache")
    datasets_dir = str(DATA_DIR / "datasets")
    models_dir = DATA_DIR / "models"

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print(f"Downloading to {DATA_DIR}")
    print("=" * 60)

    print("\n--- Model weights ---")
    for short_name, (repo_id, kwargs) in MODELS.items():
        print(f"\n  {repo_id} -> models/{short_name}")
        path = snapshot_download(repo_id, cache_dir=cache_dir, **kwargs)
        print(f"    snapshot: {path}")
        symlink_model(short_name, path, models_dir)

    print("\n--- Taboo LoRA adapters ---")
    for word in TABOO_WORDS:
        repo_id = TABOO_LORA_TEMPLATE.format(word=word)
        print(f"\n  {repo_id}")
        try:
            snapshot_download(repo_id, cache_dir=cache_dir)
            print(f"    done")
        except Exception as e:
            print(f"    FAILED: {e}")

    print("\n--- Classification datasets ---")
    for hf_path, config_name, splits in HF_DATASETS:
        label = f"{hf_path}" + (f" ({config_name})" if config_name else "")
        print(f"\n  {label}")
        ds = load_dataset(hf_path, name=config_name, split=splits, cache_dir=datasets_dir)
        if isinstance(ds, dict):
            print(f"    -> splits: {list(ds.keys())}, rows: {sum(len(s) for s in ds.values())}")
        else:
            print(f"    -> rows: {len(ds)}")

    print("\n--- MD Gender Bias (funpedia) ---")
    if MD_GENDER_DIR.exists() and any(MD_GENDER_DIR.glob("*.jsonl")):
        print(f"  Already downloaded: {MD_GENDER_DIR}")
    else:
        import tarfile, io, urllib.request
        print(f"  Downloading {MD_GENDER_URL}")
        data = urllib.request.urlopen(MD_GENDER_URL).read()
        print(f"  Extracting funpedia splits...")
        MD_GENDER_DIR.mkdir(parents=True, exist_ok=True)
        tf = tarfile.open(fileobj=io.BytesIO(data))
        for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
            member = tf.getmember(f"data_to_release/funpedia/{name}")
            f = tf.extractfile(member)
            (MD_GENDER_DIR / name).write_bytes(f.read())
            print(f"    {name}")
        print(f"  Saved to {MD_GENDER_DIR}")

    print("\n--- Streaming datasets (metadata probe) ---")
    for hf_path, config_name in STREAMING_DATASETS:
        label = f"{hf_path}" + (f" ({config_name})" if config_name else "")
        print(f"\n  {label}")
        ds = load_dataset(hf_path, name=config_name, split="train", streaming=True, cache_dir=datasets_dir)
        sample = next(iter(ds))
        print(f"    -> streaming OK, sample keys: {list(sample.keys())}")

    print("\n" + "=" * 60)
    print("Done. Training script reads from:")
    print(f"  Model weights:  {models_dir}/<name>/")
    print(f"  Local CSVs:     ref_submodules/activation_oracles/datasets/classification_datasets/")
    print("=" * 60)


if __name__ == "__main__":
    main()
