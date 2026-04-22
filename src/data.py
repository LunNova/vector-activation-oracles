"""Minimal classification data pipeline for oracle training."""

import copy
import csv
import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

PLACEHOLDER = " ?"


def _seeded_rng(*key_parts) -> random.Random:
    """Create a per-item deterministic RNG from a composite key.

    Python 3.13 removed tuple seed support from random.Random.
    We convert the key parts to a canonical string representation.
    """
    return random.Random(repr(key_parts))


def _get_think_token_ids(tokenizer: PreTrainedTokenizer) -> list[int]:
    """Get Qwen3 no-think prefix token IDs: <think>\\n\\n</think>\\n\\n

    These are special tokens in Qwen3's vocabulary. Encode directly rather than
    probing the chat template (which doesn't emit them in tokenize=False mode).
    """
    # <think> and </think> are added_tokens — always recognized during encode
    think_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    decoded = tokenizer.decode(think_ids, skip_special_tokens=False)
    assert "<think>" in decoded and "</think>" in decoded, (
        f"Think token encoding failed: {decoded!r} from ids {think_ids}"
    )
    return think_ids


# Answer format families for training diversity.
# When answer_format_diversity is enabled, each example randomly gets one format+prefix.
# When disabled, the first format's first prefix is always used.
ANSWER_FORMATS = [
    {
        "prefixes": [
            "Answer with 'Yes' or 'No' only.",
            "Answer yes or no.",
            "Give your answer as yes or no.",
        ],
        "positive": "Yes",
        "negative": "No",
    },
    {
        "prefixes": [
            "Answer with 'True' or 'False' only.",
            "Answer true or false.",
            "Give your answer as true or false.",
        ],
        "positive": "True",
        "negative": "False",
    },
    {
        "prefixes": [
            "Answer with 'Right' or 'Wrong' only.",
            "Answer right or wrong.",
            "Give your answer as right or wrong.",
        ],
        "positive": "Right",
        "negative": "Wrong",
    },
]

# All recognized answer tokens across all formats (for eval format correctness)
ALL_VALID_ANSWERS = {
    ans.lower() for fmt in ANSWER_FORMATS for ans in (fmt["positive"], fmt["negative"])
}

# Path to AO reference classification data files
_AO_DATA = (
    Path(__file__).parent.parent
    / "ref_submodules"
    / "activation_oracles"
    / "datasets"
    / "classification_datasets"
)
_PARAPHRASES = None  # lazy-loaded


def _get_paraphrases() -> dict:
    global _PARAPHRASES
    if _PARAPHRASES is None:
        with open(_AO_DATA / "paraphrases" / "question.json") as f:
            _PARAPHRASES = json.load(f)
    return _PARAPHRASES


# Each dataset: HF path, context field, label field, question, label→answer map, splits
# question_template: if present, formatted with item fields (e.g., "{hypothesis}")
# valid_labels: if present, skip items whose label is not in this set
# loader: custom loader function name for complex datasets
CLASSIFICATION_DATASETS = {
    "sst2": {"loader": "_load_sst2"},
    "snli": {"loader": "_load_snli"},
    "language_identification": {"loader": "_load_language_id"},
    "geometry_of_truth": {"loader": "_load_geometry_of_truth"},
    "relations": {"loader": "_load_relations"},
    "ner": {"loader": "_load_ner"},
    "tense": {"loader": "_load_tense"},
    "md_gender": {"loader": "_load_md_gender"},
    "ag_news": {"loader": "_load_ag_news"},
    "singular_plural": {"loader": "_load_singular_plural"},
}


def _load_sst2(
    num_train: int,
    num_test: int,
    num_qa_per_sample: int = 1,
    data_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """SST-2 sentiment with paraphrased questions, matching reference SstDatasetLoader."""
    paraphrases = _get_paraphrases()["sst2"]  # {"positive": [...], "negative": [...]}
    ds = load_dataset("stanfordnlp/sst2")

    def format_items(items_slice, pool_name, index_offset=0):
        result = []
        for j, item in enumerate(items_slice):
            orig_idx = index_offset + j
            label_name = {0: "negative", 1: "positive"}[item["label"]]
            for qa_idx in range(num_qa_per_sample):
                qa_rng = _seeded_rng("sst2", pool_name, orig_idx, qa_idx, "qa")
                polarity = qa_rng.choice(["positive", "negative"])
                question = qa_rng.choice(paraphrases[polarity])
                answer = "Yes" if label_name == polarity else "No"
                result.append(
                    {
                        "context": item["sentence"].strip(),
                        "question": question,
                        "answer": answer,
                        "dataset": "sst2",
                        "original_index": orig_idx,
                        "cache_group": f"cls_sst2_{pool_name}",
                    }
                )
        return result

    train_items = list(ds["train"])[:num_train]
    test_items = list(ds["validation"])[:num_test]
    return format_items(train_items, "train"), format_items(test_items, "validation")


def _load_snli(
    num_train: int,
    num_test: int,
    num_qa_per_sample: int = 1,
    data_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """SNLI entailment with paraphrased questions, matching reference SnliDatasetLoader."""
    paraphrases = _get_paraphrases()["snli"]  # flat list
    ds = load_dataset("stanfordnlp/snli")

    def format_items(split_items, pool_name, num_contexts):
        """Iterate split in order, skip invalid labels, stop after num_contexts valid items."""
        result = []
        valid_count = 0
        for j, item in enumerate(split_items):
            if item["label"] not in (0, 2):
                continue
            if valid_count >= num_contexts:
                break
            valid_count += 1
            orig_idx = j  # position in the full HF split (including filtered items)
            answer = {0: "Yes", 2: "No"}[item["label"]]
            for qa_idx in range(num_qa_per_sample):
                qa_rng = _seeded_rng("snli", pool_name, orig_idx, qa_idx, "qa")
                template = qa_rng.choice(paraphrases)
                question = f"{template} {item['hypothesis']}"
                result.append(
                    {
                        "context": item["premise"],
                        "question": question,
                        "answer": answer,
                        "dataset": "snli",
                        "original_index": orig_idx,
                        "cache_group": f"cls_snli_{pool_name}",
                    }
                )
        return result

    return format_items(ds["train"], "train", num_train), format_items(
        ds["test"], "test", num_test
    )


@dataclass
class SharedContext:
    """Target-model context shared across OracleExamples at different activation layers.

    Created once per raw example. Multiple OracleExamples (one per activation_layer)
    reference the same SharedContext, so precompute_activations can collect all layers
    in a single forward pass through the target model.

    `context_ids` is stored as int32 ndarray (~7x smaller than list[int]); accepts
    list[int] at construction and auto-converts. Set to None by precompute_activations
    once activations are extracted — no consumer reads it post-precompute.
    """

    context_ids: np.ndarray | None

    def __post_init__(self):
        if self.context_ids is not None and not isinstance(
            self.context_ids, np.ndarray
        ):
            self.context_ids = np.asarray(self.context_ids, dtype=np.int32)


@dataclass
class OracleExample:
    """A single oracle training/eval example."""

    # input_ids / labels stored as int32 ndarray (~7x smaller than list[int]).
    # Accept list[int] at construction; __post_init__ converts.
    input_ids: np.ndarray  # full prompt + answer tokens (includes "Layer: X")
    labels: np.ndarray  # -100 for prompt, token ids for answer
    injection_positions: list[int]  # positions of placeholder tokens
    context: SharedContext  # shared target-model context
    context_positions: list[int]  # positions in context to collect activations from
    activation_layer: int
    answer: str  # ground truth text ("Yes" / "No")
    dataset_name: str
    original_index: int = 0  # stable index within dataset pool (for cache keying)
    cache_group: str = (
        ""  # cache directory name including hash (e.g. "cls_sst2_train_a1b2c3")
    )
    cached_activations: torch.Tensor | None = (
        None  # pre-computed (K, D), set by train.py
    )
    _from_cache: bool = False  # True if loaded from disk cache (vs freshly computed)

    def __post_init__(self):
        if not isinstance(self.input_ids, np.ndarray):
            self.input_ids = np.asarray(self.input_ids, dtype=np.int32)
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.asarray(self.labels, dtype=np.int32)


def _load_language_id(
    num_train: int,
    num_test: int,
    num_qa_per_sample: int = 1,
    data_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Language identification: balanced yes/no by randomly asking about correct or incorrect language."""
    paraphrases = _get_paraphrases()["language_identification"]
    ds = load_dataset("FrancophonIA/WiLI-2018")["train"]
    items = list(ds)  # original order, no shuffle
    all_languages = sorted(
        {item["language"] for item in items}
    )  # sorted for determinism

    def format_items(items_slice, index_offset=0):
        result = []
        for j, item in enumerate(items_slice):
            orig_idx = index_offset + j
            correct = item["language"]
            for qa_idx in range(num_qa_per_sample):
                qa_rng = _seeded_rng("language_identification", orig_idx, qa_idx, "qa")
                question_template = qa_rng.choice(paraphrases)
                if qa_rng.random() < 0.5:
                    language, answer = correct, "Yes"
                else:
                    language = qa_rng.choice(
                        [lang for lang in all_languages if lang != correct]
                    )
                    answer = "No"
                result.append(
                    {
                        "context": item["Text"],
                        "question": question_template.format(language),
                        "answer": answer,
                        "dataset": "language_identification",
                        "original_index": orig_idx,
                        "cache_group": "cls_language_identification",
                    }
                )
        return result

    total = num_train + num_test
    return format_items(items[:num_train]), format_items(
        items[num_train:total], index_offset=num_train
    )


def _load_geometry_of_truth(
    num_train: int,
    num_test: int,
    num_qa_per_sample: int = 1,
    data_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Geometry of truth: true/false factual statements from multiple CSV files."""
    paraphrases = _get_paraphrases()["geometry_of_truth"]
    gmt_dir = _AO_DATA / "gmt"
    csv_names = [
        "sp_en_trans",
        "cities",
        "smaller_than",
        "larger_than",
        "companies_true_false",
    ]

    raw_contexts = []
    for name in csv_names:
        with open(gmt_dir / f"{name}.csv") as f:
            for row in csv.DictReader(f):
                raw_contexts.append(
                    (row["statement"], {"0": "No", "1": "Yes"}[row["label"]])
                )

    # No shuffle — deterministic order from fixed csv_names list
    train_contexts = raw_contexts[:num_train]
    test_contexts = raw_contexts[num_train : num_train + num_test]

    def expand(contexts, index_offset=0):
        items = []
        for j, (statement, answer) in enumerate(contexts):
            orig_idx = index_offset + j
            for qa_idx in range(num_qa_per_sample):
                qa_rng = _seeded_rng("geometry_of_truth", orig_idx, qa_idx, "qa")
                items.append(
                    {
                        "context": statement,
                        "question": qa_rng.choice(paraphrases),
                        "answer": answer,
                        "dataset": "geometry_of_truth",
                        "original_index": orig_idx,
                        "cache_group": "cls_geometry_of_truth",
                    }
                )
        return items

    return expand(train_contexts), expand(test_contexts, index_offset=num_train)


def _load_relations(
    num_train: int,
    num_test: int,
    num_qa_per_sample: int = 1,
    data_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Relations: factual/commonsense/linguistic relational knowledge."""
    paraphrases = _get_paraphrases()["relations"]
    rel_dir = _AO_DATA / "relations"

    # Collect raw contexts (each relation sample produces true + false pair).
    # Context text depends on RNG (template choice, false object) — use per-item RNG
    # keyed on stable file/sample identity so context is deterministic.
    raw_contexts = []
    for subdir in ["factual", "commonsense", "linguistic"]:
        subpath = rel_dir / subdir
        if not subpath.is_dir():
            continue
        for json_file in sorted(
            subpath.glob("*.json")
        ):  # sorted for deterministic order
            with open(json_file) as f:
                data = json.load(f)
            templates = data["prompt_templates"]
            objects = sorted(
                {s["object"] for s in data["samples"]}
            )  # sorted for determinism
            for sample_idx, sample in enumerate(data["samples"]):
                ctx_rng = _seeded_rng("relations", subdir, json_file.name, sample_idx)
                template = ctx_rng.choice(templates) + " {}."
                # True example
                context_true = template.format(sample["subject"], sample["object"])
                raw_contexts.append((context_true, "Yes"))
                # False example with random wrong object
                false_obj = ctx_rng.choice(
                    [o for o in objects if o != sample["object"]]
                )
                context_false = template.format(sample["subject"], false_obj)
                raw_contexts.append((context_false, "No"))

    # No shuffle — deterministic file iteration order. Train first.
    train_contexts = raw_contexts[:num_train]
    test_contexts = raw_contexts[num_train : num_train + num_test]

    def expand(contexts, index_offset=0):
        items = []
        for j, (context, answer) in enumerate(contexts):
            orig_idx = index_offset + j
            for qa_idx in range(num_qa_per_sample):
                qa_rng = _seeded_rng("relations", orig_idx, qa_idx, "qa")
                items.append(
                    {
                        "context": context,
                        "question": qa_rng.choice(paraphrases),
                        "answer": answer,
                        "dataset": "relations",
                        "original_index": orig_idx,
                        "cache_group": "cls_relations",
                    }
                )
        return items

    return expand(train_contexts), expand(test_contexts, index_offset=num_train)


def _load_ner(
    num_train: int,
    num_test: int,
    num_qa_per_sample: int = 1,
    data_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """NER: does this text mention a specific named entity?"""
    paraphrases = _get_paraphrases()["ner"]

    # Parse CoNLL-style NER CSV
    sentences = []
    all_entities = set()
    with open(_AO_DATA / "ner" / "ner.csv", encoding="unicode_escape") as f:
        reader = csv.DictReader(f)
        current_sentence, sentence_entities, current_entity = [], [], []
        for row in reader:
            if row["Sentence #"].strip() and current_sentence:
                if current_entity:
                    entity = " ".join(current_entity)
                    sentence_entities.append(entity)
                    all_entities.add(entity)
                    current_entity = []
                sentences.append((current_sentence, sentence_entities))
                current_sentence, sentence_entities = [], []
            current_sentence.append(row["Word"])
            tag = row["Tag"]
            if (tag == "O" or tag.startswith("B")) and current_entity:
                entity = " ".join(current_entity)
                all_entities.add(entity)
                sentence_entities.append(entity)
                current_entity = []
            if tag.startswith("B"):
                current_entity = [row["Word"]]
            elif tag.startswith("I"):
                current_entity.append(row["Word"])
        if current_sentence:
            if current_entity:
                entity = " ".join(current_entity)
                all_entities.add(entity)
                sentence_entities.append(entity)
            sentences.append((current_sentence, sentence_entities))

    all_entities_list = sorted(all_entities)  # sorted for determinism
    raw_contexts = [(words, entities) for words, entities in sentences if entities]
    # No shuffle — original CSV order
    train_contexts = raw_contexts[:num_train]
    test_contexts = raw_contexts[num_train : num_train + num_test]

    def expand(contexts, index_offset=0):
        items = []
        for j, (words, entities) in enumerate(contexts):
            orig_idx = index_offset + j
            context = " ".join(words)
            for qa_idx in range(num_qa_per_sample):
                qa_rng = _seeded_rng("ner", orig_idx, qa_idx, "qa")
                entity = qa_rng.choice(entities)
                if qa_rng.random() < 0.5:
                    question = qa_rng.choice(paraphrases).format(entity)
                    items.append(
                        {
                            "context": context,
                            "question": question,
                            "answer": "Yes",
                            "dataset": "ner",
                            "original_index": orig_idx,
                            "cache_group": "cls_ner",
                        }
                    )
                else:
                    wrong = entity
                    while wrong in set(entities):
                        wrong = qa_rng.choice(all_entities_list)
                    question = qa_rng.choice(paraphrases).format(wrong)
                    items.append(
                        {
                            "context": context,
                            "question": question,
                            "answer": "No",
                            "dataset": "ner",
                            "original_index": orig_idx,
                            "cache_group": "cls_ner",
                        }
                    )
        return items

    return expand(train_contexts), expand(test_contexts, index_offset=num_train)


def _load_tense(
    num_train: int,
    num_test: int,
    num_qa_per_sample: int = 1,
    data_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Tense: is this sentence in {past/present/future} tense?"""
    paraphrases = _get_paraphrases()["tense"]

    with open(_AO_DATA / "tense" / "tense_processed.json") as f:
        examples = json.load(f)

    tenses = sorted({ex["label"] for ex in examples})  # sorted for determinism
    # No shuffle — original JSON order
    train_contexts = examples[:num_train]
    test_contexts = examples[num_train : num_train + num_test]

    def expand(contexts, index_offset=0):
        items = []
        for j, ex in enumerate(contexts):
            orig_idx = index_offset + j
            correct = ex["label"]
            for qa_idx in range(num_qa_per_sample):
                qa_rng = _seeded_rng("tense", orig_idx, qa_idx, "qa")
                if qa_rng.random() < 0.5:
                    tense, answer = correct, "Yes"
                else:
                    tense = qa_rng.choice([t for t in tenses if t != correct])
                    answer = "No"
                question = qa_rng.choice(paraphrases).format(tense)
                items.append(
                    {
                        "context": ex["sentence"],
                        "question": question,
                        "answer": answer,
                        "dataset": "tense",
                        "original_index": orig_idx,
                        "cache_group": "cls_tense",
                    }
                )
        return items

    return expand(train_contexts), expand(test_contexts, index_offset=num_train)


def _load_md_gender(
    num_train: int,
    num_test: int,
    num_qa_per_sample: int = 1,
    data_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """MD Gender Bias: is this text about a female/male person?

    Loads from pre-downloaded JSONL files (see scripts/download_assets.py).
    Requires data_dir to locate the external data directory.
    """
    if data_dir is None:
        raise ValueError("md_gender requires data_dir (pass cfg.data_dir)")
    md_gender_dir = Path(data_dir) / "datasets" / "md_gender_funpedia"

    paraphrases = _get_paraphrases()["md_gender"]

    raw = []
    for split_name in ("train", "valid", "test"):
        jsonl_path = md_gender_dir / f"{split_name}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"MD Gender data not found at {jsonl_path}. "
                "Run scripts/download_assets.py first."
            )
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                gender = item["gender"]
                if gender not in ("male", "female"):
                    continue
                raw.append((item["text"], item["title"], gender))

    # Balance deterministically: cap male count to female count in original order
    female_count = sum(1 for _, _, g in raw if g == "female")
    male_seen = 0
    balanced = []
    for text, entity, gender in raw:
        if gender == "male":
            if male_seen >= female_count:
                continue
            male_seen += 1
        balanced.append((text, entity, gender))

    # No shuffle — original JSONL file order, balanced
    train_contexts = balanced[:num_train]
    test_contexts = balanced[num_train : num_train + num_test]

    def expand(contexts, index_offset=0):
        items = []
        for j, (text, entity, gender) in enumerate(contexts):
            orig_idx = index_offset + j
            context = f"{text}\n\nThis text is about {entity}."
            for qa_idx in range(num_qa_per_sample):
                qa_rng = _seeded_rng("md_gender", orig_idx, qa_idx, "qa")
                if qa_rng.random() < 0.5:
                    ask_gender, answer = gender, "Yes"
                else:
                    ask_gender = "female" if gender == "male" else "male"
                    answer = "No"
                question = qa_rng.choice(paraphrases).format(ask_gender)
                items.append(
                    {
                        "context": context,
                        "question": question,
                        "answer": answer,
                        "dataset": "md_gender",
                        "original_index": orig_idx,
                        "cache_group": "cls_md_gender",
                    }
                )
        return items

    return expand(train_contexts), expand(test_contexts, index_offset=num_train)


def _load_ag_news(
    num_train: int,
    num_test: int,
    num_qa_per_sample: int = 1,
    data_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """AG News: is this article about {topic}?"""
    paraphrases = _get_paraphrases()["ag_news"]
    label_to_topic = {
        "1": "World News",
        "2": "Sports",
        "3": "Business",
        "4": "Science/Technology",
    }
    labels = sorted(label_to_topic.keys())  # sorted for determinism

    raw_contexts = []
    with open(_AO_DATA / "ag_news" / "ag_news.csv") as f:
        for row in csv.DictReader(f):
            context = f"{row['Title']}\n\n{row['Description']}"
            raw_contexts.append((context, row["Class Index"]))

    # No shuffle — original CSV order
    train_contexts = raw_contexts[:num_train]
    test_contexts = raw_contexts[num_train : num_train + num_test]

    def expand(contexts, index_offset=0):
        items = []
        for j, (context, correct) in enumerate(contexts):
            orig_idx = index_offset + j
            for qa_idx in range(num_qa_per_sample):
                qa_rng = _seeded_rng("ag_news", orig_idx, qa_idx, "qa")
                if qa_rng.random() < 0.5:
                    topic, answer = label_to_topic[correct], "Yes"
                else:
                    wrong = qa_rng.choice([lbl for lbl in labels if lbl != correct])
                    topic, answer = label_to_topic[wrong], "No"
                question = qa_rng.choice(paraphrases).format(topic)
                items.append(
                    {
                        "context": context,
                        "question": question,
                        "answer": answer,
                        "dataset": "ag_news",
                        "original_index": orig_idx,
                        "cache_group": "cls_ag_news",
                    }
                )
        return items

    return expand(train_contexts), expand(test_contexts, index_offset=num_train)


def _load_singular_plural(
    num_train: int,
    num_test: int,
    num_qa_per_sample: int = 1,
    data_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Singular/plural: does this sentence refer to single/multiple people?"""
    paraphrases = _get_paraphrases()["singular_plural"]
    class_labels = sorted({"single", "multiple"})  # sorted for determinism

    rows = []
    with open(_AO_DATA / "singular_plural" / "singular_plural.csv") as f:
        for row in csv.DictReader(f):
            rows.append((row["sentence"].strip(), row["n_subjects"].strip()))

    # No shuffle — original CSV order
    train_rows = rows[:num_train]
    test_rows = rows[num_train : num_train + num_test]

    def expand(contexts, index_offset=0):
        items = []
        for j, (sentence, correct_label) in enumerate(contexts):
            orig_idx = index_offset + j
            for qa_idx in range(num_qa_per_sample):
                qa_rng = _seeded_rng("singular_plural", orig_idx, qa_idx, "qa")
                if qa_rng.random() < 0.5:
                    ask_label, answer = correct_label, "Yes"
                else:
                    ask_label = qa_rng.choice(
                        [lbl for lbl in class_labels if lbl != correct_label]
                    )
                    answer = "No"
                question = qa_rng.choice(paraphrases[ask_label]).format(ask_label)
                items.append(
                    {
                        "context": sentence,
                        "question": question,
                        "answer": answer,
                        "dataset": "singular_plural",
                        "original_index": orig_idx,
                        "cache_group": "cls_singular_plural",
                    }
                )
        return items

    return expand(train_rows), expand(test_rows, index_offset=num_train)


def load_classification_data(
    dataset_name: str,
    num_train: int,
    num_test: int,
    num_qa_per_sample: int = 1,
    data_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load and format a classification dataset.

    num_qa_per_sample: generate N question/answer pairs per context using paraphrases.
    num_train/num_test refer to context counts; returned list has N*count items.
    data_dir: path to data directory (needed by md_gender which loads from external data).
    All RNG is per-item (keyed on stable identifiers), not sequential.
    Raw dicts have "original_index" and "cache_group" keys for cache keying.
    """
    cfg = CLASSIFICATION_DATASETS[dataset_name]
    loader_fn = globals()[cfg["loader"]]
    return loader_fn(
        num_train, num_test, num_qa_per_sample=num_qa_per_sample, data_dir=data_dir
    )


def find_placeholder_positions(
    token_ids: list[int], placeholder_id: int, expected: int
) -> list[int]:
    """Find the first `expected` consecutive placeholder tokens in token_ids.

    Matches AO reference find_pattern_in_tokens: stops after finding N,
    asserts they're consecutive, ignores any later occurrences (e.g., "?"
    in the question text that tokenizes as the same ID).
    """
    positions = []
    for i, t in enumerate(token_ids):
        if t == placeholder_id:
            positions.append(i)
            if len(positions) == expected:
                break
    if len(positions) != expected:
        raise ValueError(
            f"Expected {expected} placeholder tokens, found {len(positions)}. "
            f"Token ID {placeholder_id} may not be the right placeholder."
        )
    if positions[-1] - positions[0] != expected - 1:
        raise ValueError(f"Placeholder tokens are not consecutive: {positions}")
    return positions


def prepare_examples(
    raw_examples: list[dict],
    tokenizer: PreTrainedTokenizer,
    activation_layers: list[int],
    min_k_activations: int,
    max_k_activations: int,
    max_context_len: int,
    answer_format_diversity: bool,
    supervise_think_tokens: bool,
    min_end_offset: int = -1,
    max_end_offset: int = -5,
) -> list[OracleExample]:
    """Convert raw classification examples to oracle training format.

    Each example is expanded across all activation_layers (one OracleExample per layer),
    matching AO reference (classification.py:224 inner loop over act_layers).
    num_positions is randomly sampled from [min_k_activations, max_k_activations]
    per example (ref: classification.py:235).
    end_offset is randomized from [max_end_offset, min_end_offset] (both negative),
    shifting the activation window away from the very end of the context
    (ref: classification.py:229).
    If answer_format_diversity is True, randomly varies the answer format
    (Yes/No, True/False, Right/Wrong) and prefix per example.

    Raw examples must have "original_index" and "cache_group" keys set by the loader.
    All RNG is per-item (keyed on stable identifiers), not sequential.
    """
    placeholder_ids = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
    assert len(placeholder_ids) == 1, (
        f"Expected '{PLACEHOLDER}' to be a single token, got {len(placeholder_ids)} tokens: "
        f"{placeholder_ids} -> {[tokenizer.decode([t]) for t in placeholder_ids]}"
    )
    placeholder_id = placeholder_ids[0]

    # Qwen3 no-think prefix: <think>\n\n</think>\n\n (supervised, matching AO reference)
    think_token_ids = _get_think_token_ids(tokenizer)

    # Reuse SharedContext across QA pairs from the same original item
    context_cache: dict[int, SharedContext] = {}

    prepared = []
    for i, ex in enumerate(raw_examples):
        dataset_name = ex["dataset"]
        orig_idx = ex.get("original_index", i)
        cache_group = ex.get("cache_group", "")

        # Tokenize context once per unique original item.
        # Chat-format first, matching AO reference (classification.py:207-208):
        # the target model sees chat-formatted input during activation collection.
        if orig_idx not in context_cache:
            context_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["context"]}],
                tokenize=False,
                enable_thinking=False,
            )
            context_ids = tokenizer.encode(
                context_str,
                add_special_tokens=False,
                max_length=max_context_len,
                truncation=True,
            )
            context_cache[orig_idx] = SharedContext(context_ids=context_ids)
        context = context_cache[orig_idx]
        ctx_len = len(context.context_ids)

        # Prompt-only RNG — safe to have conditional calls (doesn't affect activations)
        # enumerate index `i` distinguishes QA pairs from the same original item
        prompt_rng = _seeded_rng(dataset_name, orig_idx, i, "prompt")
        question = ex["question"]
        answer = ex["answer"]
        if answer_format_diversity:
            fmt = prompt_rng.choice(ANSWER_FORMATS)
            prefix = prompt_rng.choice(fmt["prefixes"])
            if answer == "Yes":
                answer = fmt["positive"]
            elif answer == "No":
                answer = fmt["negative"]
        else:
            prefix = ANSWER_FORMATS[0]["prefixes"][0]
        question = f"{prefix} {question}"

        for activation_layer in activation_layers:
            # Activation RNG — fixed call sequence, no conditionals.
            # Includes (min_k, max_k) so different classification variants get
            # independent streams (randint(1,1) consumes 0 bits, would corrupt
            # subsequent draws if sharing state with a wider-range variant).
            act_rng = _seeded_rng(
                dataset_name,
                orig_idx,
                activation_layer,
                min_k_activations,
                max_k_activations,
            )
            num_positions = act_rng.randint(min_k_activations, max_k_activations)
            end_offset = act_rng.randint(max_end_offset, min_end_offset)

            end_pos = ctx_len + end_offset  # inclusive last position
            if end_pos < 1:
                continue  # context too short for this offset
            num_pos = min(num_positions, end_pos + 1)
            context_positions = list(range(end_pos - num_pos + 1, end_pos + 1))

            # Oracle prompt matching AO paper format:
            # "Layer: {L}\n{placeholders} \n{question}"
            # Trailing " \n" prevents last " ?" from merging with newline in BPE
            # Use num_pos (not num_positions) so placeholder count matches activations.
            placeholders = PLACEHOLDER * num_pos
            prompt_content = f"Layer: {activation_layer}\n{placeholders} \n{question}"

            # Tokenize prompt via chat template, answer separately, then concatenate.
            user_messages = [{"role": "user", "content": prompt_content}]
            prompt_str = tokenizer.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)

            # prompt_ids already ends with <think>\n\n</think>\n\n from Qwen3's
            # no-think generation prompt — find the boundary for label masking.
            think_len = len(think_token_ids)
            assert prompt_ids[-think_len:] == think_token_ids, (
                f"Expected prompt to end with think tokens, got "
                f"{prompt_ids[-think_len:]} vs {think_token_ids}"
            )
            prompt_before_think = prompt_ids[:-think_len]

            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            full_ids = prompt_ids + answer_ids + [im_end_id]
            think_labels = (
                think_token_ids if supervise_think_tokens else [-100] * think_len
            )
            labels = (
                [-100] * len(prompt_before_think)
                + think_labels
                + answer_ids
                + [im_end_id]
            )

            injection_positions = find_placeholder_positions(
                full_ids, placeholder_id, num_pos
            )

            prepared.append(
                OracleExample(
                    input_ids=full_ids,
                    labels=labels,
                    injection_positions=injection_positions,
                    context=context,
                    context_positions=context_positions,
                    activation_layer=activation_layer,
                    answer=answer,
                    dataset_name=dataset_name,
                    original_index=orig_idx,
                    cache_group=cache_group,
                )
            )

    return prepared


# --- Context prediction data pipeline ---


def _stream_mixed_texts(
    tokenizer: PreTrainedTokenizer,
    pretrain_frac: float,
):
    """Generator yielding raw text strings from FineWeb + LMSYS Chat-1M.

    Alternates between pretrain and chat at the specified ratio.
    Matches AO reference hf_mixed_dataset_to_generator (past_lens_dataset.py:72-166).
    """
    pretrain_ds = iter(
        load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
    )

    try:
        chat_ds = iter(
            load_dataset(
                "lmsys/lmsys-chat-1m",
                split="train",
                streaming=True,
            )
        )
        has_chat = True
    except Exception as e:
        print(f"Warning: LMSYS Chat-1M unavailable ({e}), using FineWeb only")
        has_chat = False

    # Convert fraction to integer ratio (e.g., 0.5 → 1:1)
    from fractions import Fraction

    frac = Fraction(pretrain_frac).limit_denominator(10)
    n_pretrain = frac.numerator
    n_chat = frac.denominator - n_pretrain if has_chat else 0

    while True:
        for _ in range(n_pretrain):
            sample = next(pretrain_ds)
            yield sample["text"]
        for _ in range(n_chat):
            sample = next(chat_ds)
            # Apply chat template to get flat text string
            text = tokenizer.apply_chat_template(
                sample["conversation"],
                tokenize=False,
                enable_thinking=False,
            )
            yield text


def load_context_prediction_data(
    tokenizer: PreTrainedTokenizer,
    num_examples: int,
    max_context_len: int,
    min_k_tokens: int,
    max_k_tokens: int,
    min_k_activations: int,
    max_k_activations: int,
    pretrain_frac: float = 0.5,
    variant_idx: int = 0,
) -> list[dict]:
    """Stream FineWeb + LMSYS, tokenize, and extract context prediction examples.

    Returns list of dicts with pre-tokenized context fields and variable num_positions.
    Matches AO reference collect_past_lens_acts (past_lens_dataset.py:169-295).

    All RNG is per-stream-position (keyed on stable identifiers), not sequential.
    Raw dicts have "original_index" and "cache_group" keys for cache keying.
    """
    stream = _stream_mixed_texts(tokenizer, pretrain_frac)
    examples = []
    stream_idx = 0

    while len(examples) < num_examples:
        text = next(stream)
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=max_context_len,
            truncation=True,
        )
        L = len(token_ids)

        # Per-item RNG keyed on stream position — deterministic regardless of seed
        item_rng = _seeded_rng("context_prediction", variant_idx, stream_idx)
        stream_idx += 1

        k_acts = item_rng.randint(min_k_activations, max_k_activations)
        k_tokens = item_rng.randint(min_k_tokens, max_k_tokens)

        # Need at least k_tokens + k_acts + 1 tokens
        if L < k_tokens + k_acts + 1:
            continue

        direction = item_rng.choice(["past", "future"])

        if direction == "past":
            # Activation window must have k_tokens before it and >= 1 token after
            act_begin_min = k_tokens
            act_begin_max = L - k_acts - 1
            if act_begin_max < act_begin_min:
                continue
            act_begin = item_rng.randint(act_begin_min, act_begin_max)
            act_positions = list(range(act_begin, act_begin + k_acts))
            target_positions = list(range(act_begin - k_tokens, act_begin))
            context_cutoff = act_positions[-1]
            question = (
                f"Can you predict the previous {k_tokens} tokens that came before this?"
            )
        else:  # future
            act_begin_min = 1
            act_begin_max = L - k_acts - k_tokens
            if act_begin_max < act_begin_min:
                continue
            act_begin = item_rng.randint(act_begin_min, act_begin_max)
            act_positions = list(range(act_begin, act_begin + k_acts))
            last_act = act_positions[-1]
            target_positions = list(range(last_act + 1, last_act + 1 + k_tokens))
            context_cutoff = last_act
            question = (
                f"Can you predict the next {k_tokens} tokens that come after this?"
            )

        # Decode target tokens to text
        target_token_ids = [token_ids[p] for p in target_positions]
        target_text = tokenizer.decode(target_token_ids, skip_special_tokens=True)

        # Context for activation collection: up through last activation position
        context_ids = token_ids[: context_cutoff + 1]

        examples.append(
            {
                "context_ids": context_ids,
                "context_positions": act_positions,
                "question": question,
                "answer": target_text,
                "dataset": "context_prediction",
                "num_positions": k_acts,
                "original_index": stream_idx - 1,
                "cache_group": f"cp_{variant_idx}",
            }
        )

        if len(examples) % 10000 == 0:
            print(f"  Context prediction: {len(examples)}/{num_examples}", flush=True)

    return examples


def prepare_context_prediction_examples(
    raw_examples: list[dict],
    tokenizer: PreTrainedTokenizer,
    activation_layers: list[int],
    max_answer_tokens: int | None,
    max_num_positions: int | None,
    supervise_think_tokens: bool,
) -> list[OracleExample]:
    """Convert context prediction raw examples to OracleExample format.

    Each example is expanded across all activation_layers (one OracleExample per layer),
    matching AO reference (past_lens_dataset.py:215 inner loop over layers).
    Accepts pre-tokenized context_ids/context_positions and variable num_positions.
    Raw examples must have "original_index" and "cache_group" keys.
    """
    return _prepare_variable_position_examples(
        raw_examples,
        tokenizer,
        activation_layers,
        max_answer_tokens,
        max_num_positions,
        supervise_think_tokens,
        layer_mode="expand",
    )


def _prepare_variable_position_examples(
    raw_examples: list[dict],
    tokenizer: PreTrainedTokenizer,
    activation_layers: list[int],
    max_answer_tokens: int | None,
    max_num_positions: int | None,
    supervise_think_tokens: bool,
    layer_mode: str = "expand",  # "expand" = iterate all layers, "random" = rng.choice
) -> list[OracleExample]:
    """Shared implementation for context prediction and SPQA example preparation.

    All RNG is per-item (keyed on stable identifiers), not sequential.
    """
    placeholder_ids = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
    assert len(placeholder_ids) == 1, (
        f"Expected '{PLACEHOLDER}' to be a single token, got {len(placeholder_ids)}"
    )
    placeholder_id = placeholder_ids[0]

    think_token_ids = _get_think_token_ids(tokenizer)

    prepared = []
    filtered_answer = 0
    filtered_positions = 0

    for ex in raw_examples:
        answer_ids = tokenizer.encode(ex["answer"], add_special_tokens=False)
        if max_answer_tokens is not None and len(answer_ids) >= max_answer_tokens:
            filtered_answer += 1
            continue
        if max_num_positions is not None and ex["num_positions"] >= max_num_positions:
            filtered_positions += 1
            continue

        orig_idx = ex.get("original_index", 0)
        cache_group = ex.get("cache_group", "")

        # Shared context — one per raw example, reused across layer expansions
        context = SharedContext(context_ids=ex["context_ids"])

        if layer_mode == "expand":
            layers_iter = activation_layers
        else:
            # Per-item layer RNG keyed on stable identifier
            layer_rng = _seeded_rng(ex["dataset"], orig_idx, "layer_choice")
            layers_iter = [layer_rng.choice(activation_layers)]

        for activation_layer in layers_iter:
            num_positions = ex["num_positions"]

            placeholders = PLACEHOLDER * num_positions
            prompt_content = (
                f"Layer: {activation_layer}\n{placeholders} \n{ex['question']}"
            )

            user_messages = [{"role": "user", "content": prompt_content}]
            prompt_str = tokenizer.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

            # prompt_ids already ends with <think>\n\n</think>\n\n from Qwen3's
            # no-think generation prompt — find the boundary for label masking.
            think_len = len(think_token_ids)
            assert prompt_ids[-think_len:] == think_token_ids, (
                f"Expected prompt to end with think tokens, got "
                f"{prompt_ids[-think_len:]} vs {think_token_ids}"
            )
            prompt_before_think = prompt_ids[:-think_len]

            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            full_ids = prompt_ids + answer_ids + [im_end_id]
            think_labels = (
                think_token_ids if supervise_think_tokens else [-100] * think_len
            )
            labels = (
                [-100] * len(prompt_before_think)
                + think_labels
                + answer_ids
                + [im_end_id]
            )

            injection_positions = find_placeholder_positions(
                full_ids, placeholder_id, num_positions
            )

            prepared.append(
                OracleExample(
                    input_ids=full_ids,
                    labels=labels,
                    injection_positions=injection_positions,
                    context=context,
                    context_positions=ex["context_positions"],
                    activation_layer=activation_layer,
                    answer=ex["answer"],
                    dataset_name=ex["dataset"],
                    original_index=orig_idx,
                    cache_group=cache_group,
                )
            )

    if filtered_answer:
        print(
            f"  Filtered {filtered_answer}/{len(raw_examples)} examples with answers >= {max_answer_tokens} tokens"
        )
    if filtered_positions:
        print(
            f"  Filtered {filtered_positions}/{len(raw_examples)} examples with >= {max_num_positions} positions"
        )
    return prepared


# --- SPQA (System Prompt QA) data pipeline ---

# Path to latentqa loader in ref_submodules (standalone, zero dependencies)
_LATENTQA_LOADER = (
    Path(__file__).parent.parent
    / "ref_submodules"
    / "activation_oracles"
    / "nl_probes"
    / "dataset_classes"
    / "misc"
)
_LATENTQA_DATA = (
    Path(__file__).parent.parent
    / "ref_submodules"
    / "activation_oracles"
    / "datasets"
    / "latentqa_datasets"
    / "train"
)


def load_spqa_data(
    tokenizer: PreTrainedTokenizer,
    num_examples: int,
    min_window_size: int = 1,
    max_window_size: int = 3,
) -> list[dict]:
    """Load SPQA (system prompt QA) data from LatentQA JSON files.

    Each example is a conversation (read_prompt) paired with a QA about its content.
    Context for activation collection is the tokenized conversation.
    Matches AO reference LatentQADatasetLoader (latentqa_dataset.py).

    Returns list of dicts with pre-tokenized context fields.
    All RNG is per-item (keyed on LatentQA dataset index), not sequential.
    Iterates in original LatentQA order (no shuffle).
    """
    import sys

    if str(_LATENTQA_LOADER) not in sys.path:
        sys.path.insert(0, str(_LATENTQA_LOADER))
    import latentqa_loader

    paths = latentqa_loader.DataPaths(
        system=None,  # system.json not used in AO reference default config
        stimulus_completion=str(_LATENTQA_DATA / "stimulus_completion.json"),
        stimulus=str(_LATENTQA_DATA / "stimulus.json"),
        control=str(_LATENTQA_DATA / "control.json"),
        qa=str(_LATENTQA_DATA / "qa.json"),
    )
    ds = latentqa_loader.load_latentqa_dataset(
        paths,
        filter_prefixes=[],
        train_percent=1.0,
        add_thought_tokens=False,
        seed=0,  # seed unused at train_percent=1.0
    )
    print(f"  SPQA: loaded {len(ds)} raw datapoints from LatentQA")

    # Masked turn counts per source (matching AO reference)
    masked_turn_count = {"stimulus_completion": 2, "stimulus": 2, "control": 0}
    examples = []

    # Iterate in original LatentQA order (no shuffle) — deterministic
    for idx in range(len(ds)):
        if len(examples) >= num_examples:
            break
        item = ds[idx]

        read_prompt = item["read_prompt"]
        dialog = item["dialog"]
        source = item["source"]

        if source not in masked_turn_count:
            continue

        num_masked = masked_turn_count[source]

        # Tokenize masked turns (to compute offset for context_positions)
        masked_turns = read_prompt[:num_masked]
        if num_masked > 0:
            masked_str = tokenizer.apply_chat_template(
                masked_turns,
                tokenize=False,
                enable_thinking=False,
            )
            masked_token_count = len(
                tokenizer.encode(masked_str, add_special_tokens=False)
            )
        else:
            masked_token_count = 0

        # Tokenize full read_prompt → context_input_ids
        add_gen_prompt = source != "stimulus_completion"
        full_read_str = tokenizer.apply_chat_template(
            read_prompt,
            tokenize=False,
            add_generation_prompt=add_gen_prompt,
            enable_thinking=False,
        )
        context_input_ids = tokenizer.encode(full_read_str, add_special_tokens=False)

        # Context positions: all positions after masked turns
        all_positions = list(range(masked_token_count, len(context_input_ids)))
        if not all_positions:
            continue

        # Per-item RNG keyed on LatentQA dataset index
        item_rng = _seeded_rng("spqa", idx)

        # 50/50 "all" vs "window" position types (ref: latentqa_dataset.py:133)
        position_type = item_rng.choice(["all", "window"])

        if position_type == "all":
            context_positions = all_positions
        else:
            # Random window near end of context (ref: latentqa_dataset.py:135-147)
            window_size = item_rng.randint(min_window_size, max_window_size)
            end_offset = item_rng.randint(-10, -1)

            if abs(end_offset) > len(all_positions):
                end_offset = -len(all_positions) + 1
            window_size = min(window_size, len(all_positions) + end_offset)
            if window_size <= 0:
                continue

            window_start = end_offset - window_size
            context_positions = all_positions[window_start:end_offset]
            if not context_positions:
                continue

        examples.append(
            {
                "context_ids": context_input_ids,
                "context_positions": context_positions,
                "question": dialog[0]["content"],
                "answer": dialog[1]["content"],
                "dataset": "spqa",
                "num_positions": len(context_positions),
                "original_index": idx,
                "cache_group": "spqa",
            }
        )

    print(f"  SPQA: prepared {len(examples)} examples")
    return examples


def prepare_spqa_examples(
    raw_examples: list[dict],
    tokenizer: PreTrainedTokenizer,
    activation_layers: list[int],
    max_answer_tokens: int | None,
    max_num_positions: int | None,
    supervise_think_tokens: bool,
) -> list[OracleExample]:
    """Convert SPQA raw examples to OracleExample format.

    SPQA uses random layer assignment (not layer expansion), matching AO reference
    (latentqa_dataset.py:149: layer = random.choice(act_layers)).
    Raw examples must have "original_index" and "cache_group" keys.
    """
    return _prepare_variable_position_examples(
        raw_examples,
        tokenizer,
        activation_layers,
        max_answer_tokens,
        max_num_positions,
        supervise_think_tokens=supervise_think_tokens,
        layer_mode="random",
    )


def _hash_params(params: dict) -> str:
    """Hash activation-affecting params to 12-char hex string."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _atomic_save_safetensors(tensors: dict, path: Path, metadata: dict[str, str]):
    """Write safetensors file atomically: write to temp, os.replace."""
    from safetensors.torch import save_file as _save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        _save_file(tensors, tmp, metadata=metadata)
        os.replace(tmp, str(path))
    except BaseException:
        os.unlink(tmp)
        raise


def load_activation_cache(examples: list[OracleExample], cache_dir: str) -> int:
    """Load cached activations from disk. Returns number of cache hits.

    Cache path: {cache_dir}/{cache_group}/chunk_{original_index // 1000:04d}.safetensors
    Tensor key: "{original_index}_{activation_layer}"
    """
    from safetensors.torch import load_file

    # Group by (cache_group, chunk_index) to batch file reads
    groups: dict[tuple[str, int], list[OracleExample]] = defaultdict(list)
    for ex in examples:
        if not ex.cache_group:
            continue  # skip dummies (eval, taboo)
        chunk_idx = ex.original_index // 1000
        groups[(ex.cache_group, chunk_idx)].append(ex)

    hits = 0
    for (group, chunk_idx), exs in groups.items():
        chunk_path = Path(cache_dir) / group / f"chunk_{chunk_idx:04d}.safetensors"
        if not chunk_path.exists():
            continue
        try:
            tensors = load_file(str(chunk_path))
        except Exception:
            continue
        for ex in exs:
            key = f"{ex.original_index}_{ex.activation_layer}"
            if key in tensors:
                ex.cached_activations = tensors[key]
                ex._from_cache = True
                hits += 1

    return hits


def load_train_test_data(
    cfg,  # ExperimentConfig — not typed to avoid circular import
    tokenizer: PreTrainedTokenizer,
) -> tuple[list[OracleExample], list[OracleExample]]:
    """Load and prepare all training and test data from config.

    Parallelizes independent dataset loading tasks with a thread pool.
    Returns (all_train, all_test).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    train_ds_names = cfg.train_datasets if cfg.train_datasets else cfg.datasets
    eval_ds_names = cfg.datasets
    futures = {}

    # HuggingFace's Rust-backed tokenizers are not thread-safe (concurrent
    # .encode() calls cause "RuntimeError: Already borrowed").  Give each
    # thread-pool task its own deep copy.

    def _load_cls_train(ds_name, var_idx, variant):
        tok = copy.deepcopy(tokenizer)
        min_k = variant["min_k"]
        max_k = variant["max_k"]
        num_qa = variant.get("num_qa_per_sample", 1)
        train_raw, _ = load_classification_data(
            ds_name,
            cfg.num_train,
            cfg.num_test,
            num_qa_per_sample=num_qa,
            data_dir=cfg.data_dir,
        )
        act_hash = _hash_params(
            {
                "activation_layer_pcts": cfg.activation_layer_pcts,
                "min_k": min_k,
                "max_k": max_k,
                "max_context_len": cfg.max_context_len,
                "min_end_offset": -1,
                "max_end_offset": -5,
            }
        )
        for ex in train_raw:
            ex["cache_group"] = f"{ex['cache_group']}_{act_hash}"
        return prepare_examples(
            train_raw,
            tok,
            cfg.activation_layers,
            min_k,
            max_k,
            cfg.max_context_len,
            answer_format_diversity=cfg.answer_format_diversity,
            supervise_think_tokens=cfg.supervise_think_tokens,
        )

    def _load_cls_eval(ds_name, eval_variant):
        tok = copy.deepcopy(tokenizer)
        min_k = eval_variant["min_k"]
        max_k = eval_variant["max_k"]
        _, test_raw = load_classification_data(
            ds_name,
            cfg.num_train,
            cfg.num_test,
            data_dir=cfg.data_dir,
        )
        act_hash = _hash_params(
            {
                "activation_layer_pcts": cfg.activation_layer_pcts,
                "min_k": min_k,
                "max_k": max_k,
                "max_context_len": cfg.max_context_len,
                "min_end_offset": -1,
                "max_end_offset": -5,
            }
        )
        for ex in test_raw:
            ex["cache_group"] = f"{ex['cache_group']}_{act_hash}"
        return prepare_examples(
            test_raw,
            tok,
            cfg.activation_layers,
            min_k,
            max_k,
            cfg.max_context_len,
            answer_format_diversity=cfg.answer_format_diversity,
            supervise_think_tokens=cfg.supervise_think_tokens,
        )

    def _load_cp(var_idx, variant):
        tok = copy.deepcopy(tokenizer)
        num_raw = variant["num_raw"]
        print(
            f"Loading context prediction variant {var_idx} ({num_raw} raw examples)..."
        )
        cp_raw = load_context_prediction_data(
            tok,
            num_raw,
            max_context_len=cfg.max_context_len,
            min_k_tokens=variant["min_k_tokens"],
            max_k_tokens=variant["max_k_tokens"],
            min_k_activations=variant["min_k_acts"],
            max_k_activations=variant["max_k_acts"],
            variant_idx=var_idx,
        )
        act_hash = _hash_params(
            {
                "activation_layer_pcts": cfg.activation_layer_pcts,
                "max_context_len": cfg.max_context_len,
                "min_k_acts": variant["min_k_acts"],
                "max_k_acts": variant["max_k_acts"],
                "min_k_tokens": variant["min_k_tokens"],
                "max_k_tokens": variant["max_k_tokens"],
                "pretrain_frac": 0.5,
            }
        )
        for ex in cp_raw:
            ex["cache_group"] = f"{ex['cache_group']}_{act_hash}"
        return prepare_context_prediction_examples(
            cp_raw,
            tok,
            cfg.activation_layers,
            max_answer_tokens=cfg.max_answer_tokens,
            max_num_positions=cfg.max_num_positions,
            supervise_think_tokens=cfg.supervise_think_tokens,
        )

    def _load_spqa():
        tok = copy.deepcopy(tokenizer)
        n_spqa = 999_999 if cfg.spqa_train == -1 else cfg.spqa_train
        print(
            f"Loading SPQA data ({n_spqa if cfg.spqa_train > 0 else 'all available'} examples)..."
        )
        spqa_raw = load_spqa_data(tok, n_spqa)
        act_hash = _hash_params(
            {
                "activation_layer_pcts": cfg.activation_layer_pcts,
                "min_window_size": 1,
                "max_window_size": 3,
            }
        )
        for ex in spqa_raw:
            ex["cache_group"] = f"{ex['cache_group']}_{act_hash}"
        return prepare_spqa_examples(
            spqa_raw,
            tok,
            cfg.activation_layers,
            max_answer_tokens=cfg.max_answer_tokens,
            max_num_positions=cfg.max_num_positions,
            supervise_think_tokens=cfg.supervise_think_tokens,
        )

    print("Loading datasets...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        # Classification training — one task per (variant, dataset)
        for var_idx, variant in enumerate(cfg.classification_variants):
            for ds_name in train_ds_names:
                if ds_name not in CLASSIFICATION_DATASETS:
                    continue
                f = pool.submit(_load_cls_train, ds_name, var_idx, variant)
                futures[f] = ("train", f"cls_train/{ds_name}/v{var_idx}")

        # Classification eval — one task per dataset
        eval_variant = cfg.classification_variants[0]
        for ds_name in eval_ds_names:
            if ds_name not in CLASSIFICATION_DATASETS:
                continue
            f = pool.submit(_load_cls_eval, ds_name, eval_variant)
            futures[f] = ("test", f"cls_eval/{ds_name}")

        # Context prediction — one task per variant
        for var_idx, variant in enumerate(cfg.context_prediction_variants):
            f = pool.submit(_load_cp, var_idx, variant)
            futures[f] = ("train", f"cp/v{var_idx}")

        # SPQA
        if cfg.spqa_train != 0:
            f = pool.submit(_load_spqa)
            futures[f] = ("train", "spqa")

        # Collect results
        all_train, all_test = [], []
        for future in as_completed(futures):
            split, label = futures[future]
            examples = future.result()
            if split == "train":
                all_train.extend(examples)
            else:
                all_test.extend(examples)
            print(f"  {label}: {len(examples)} examples", flush=True)

    print(f"Train: {len(all_train)} examples, Test: {len(all_test)} examples")

    from collections import Counter

    for split_name, split_data in [("train", all_train), ("test", all_test)]:
        ds_dist = Counter(ex.dataset_name for ex in split_data)
        answer_dist = Counter(ex.answer for ex in split_data)
        print(f"  {split_name} datasets: {dict(ds_dist)}", flush=True)
        if len(answer_dist) <= 20:
            print(f"  {split_name} answers: {dict(answer_dist)}", flush=True)
        else:
            top = answer_dist.most_common(5)
            print(
                f"  {split_name} answers: {len(answer_dist)} unique ({top[0][0]!r}: {top[0][1]}, ...)",
                flush=True,
            )

    cache_dir = f"{cfg.data_dir}/activation_cache/{cfg.model_name}_{cfg.dtype}"
    all_examples = all_train + all_test
    hits = load_activation_cache(all_examples, cache_dir)
    total = len(all_examples)
    if total > 0:
        print(f"Activation cache: {hits}/{total} hits ({100 * hits / total:.0f}%)")

    return all_train, all_test


def print_dataset_summary(
    all_train: list[OracleExample],
    all_test: list[OracleExample],
    tokenizer: PreTrainedTokenizer,
    train_ds_names: set[str],
) -> None:
    """Print dataset distribution and one example per dataset for wandb log capture.

    Call after wandb.init so the output appears in the run's logs.
    """
    from collections import Counter

    print(f"\n{'=' * 60}")
    print("DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"Train: {len(all_train)}  Test: {len(all_test)}")

    for split_name, split_data in [("train", all_train), ("test", all_test)]:
        ds_dist = Counter(ex.dataset_name for ex in split_data)
        print(f"\n  {split_name} distribution:")
        for ds_name in sorted(ds_dist):
            tag = "id" if ds_name in train_ds_names else "ood"
            print(f"    {ds_name} ({tag}): {ds_dist[ds_name]}")

    seen = set()
    print(f"\n{'─' * 60}")
    print("EXAMPLE SAMPLES (1 per dataset)")
    print(f"{'─' * 60}")
    for ex in all_train + all_test:
        if ex.dataset_name in seen:
            continue
        seen.add(ex.dataset_name)
        tag = "id" if ex.dataset_name in train_ds_names else "ood"
        prompt_text = tokenizer.decode(ex.input_ids.tolist(), skip_special_tokens=False)
        if len(prompt_text) > 500:
            prompt_text = prompt_text[:500] + "..."
        print(
            f"\n[{ex.dataset_name} ({tag}), layer={ex.activation_layer}, K={len(ex.injection_positions)}]"
        )
        print(f"  prompt: {prompt_text}")
        print(f"  answer: {ex.answer!r}")
    print(f"{'=' * 60}\n", flush=True)


def collate_batch(
    examples: list[OracleExample],
    pad_token_id: int,
    device: torch.device,
) -> dict:
    """Left-pad a batch of examples for training/eval forward passes.

    Handles variable num_positions (K) across examples by padding injection_positions
    to max_K in the batch. Padded positions point to position 0; corresponding
    zero-padded activations produce zero injection via nan_to_num in _inject_activations.

    Context fields are NOT included — activation collection is handled separately
    by precompute_activations which batches by SharedContext.

    Returns dict with:
        input_ids: (B, max_len) long
        labels: (B, max_len) long
        attention_mask: (B, max_len) bool
        injection_positions: (B, max_K) long — adjusted for padding, K-padded with 0
    """
    max_len = max(len(ex.input_ids) for ex in examples)
    max_k = max(len(ex.injection_positions) for ex in examples)
    B = len(examples)

    # Build dense numpy arrays in one shot (input_ids/labels are int32 ndarrays
    # on the example; numpy slice-assign handles the left-pad).
    input_ids = np.full((B, max_len), pad_token_id, dtype=np.int64)
    labels = np.full((B, max_len), -100, dtype=np.int64)
    attention_mask = np.zeros((B, max_len), dtype=bool)
    injection_positions = np.zeros((B, max_k), dtype=np.int64)

    for i, ex in enumerate(examples):
        L = len(ex.input_ids)
        pad_len = max_len - L
        input_ids[i, pad_len:] = ex.input_ids
        labels[i, pad_len:] = ex.labels
        attention_mask[i, pad_len:] = True
        for k, p in enumerate(ex.injection_positions):
            injection_positions[i, k] = p + pad_len

    return {
        "input_ids": torch.from_numpy(input_ids).to(device),
        "labels": torch.from_numpy(labels).to(device),
        "attention_mask": torch.from_numpy(attention_mask).to(device),
        "injection_positions": torch.from_numpy(injection_positions).to(device),
    }


def stack_activations(
    examples: list[OracleExample],
    device: torch.device,
) -> torch.Tensor:
    """Stack cached activations with zero-padding for variable K.

    Each example may have a different number of activation positions (K_b).
    Pads to max_K in the batch with zeros. Zero-padded activations produce
    zero injection via nan_to_num in _inject_activations.

    Returns: (B, K_max, D) tensor.
    """
    max_k = max(ex.cached_activations.size(0) for ex in examples)
    acts = []
    for ex in examples:
        a = ex.cached_activations.to(device)
        if a.size(0) < max_k:
            a = F.pad(a, (0, 0, 0, max_k - a.size(0)))
        acts.append(a)
    return torch.stack(acts)


def _write_chunk(
    cache_dir: str,
    group: str,
    chunk_idx: int,
    new_tensors: dict[str, torch.Tensor],
) -> None:
    """Write a single chunk file. Merges with existing on-disk entries (read once)."""
    from safetensors.torch import load_file

    chunk_path = Path(cache_dir) / group / f"chunk_{chunk_idx:04d}.safetensors"
    existing = {}
    if chunk_path.exists():
        try:
            existing = load_file(str(chunk_path))
        except Exception:
            pass
    merged = {**existing, **new_tensors}
    _atomic_save_safetensors(merged, chunk_path, metadata={"cache_group": group})


def precompute_activations(
    examples: list[OracleExample],
    model,  # OracleTransformer — not typed to avoid circular import
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
    label: str = "",
    cache_dir: str | None = None,
) -> None:
    """Pre-compute and cache activations for all examples in-place.

    Groups examples by SharedContext identity so each unique context is forwarded
    through the model once, collecting activations at all needed layers in a single
    pass (via collect_activations_multi). This avoids redundant computation when
    the same context appears at multiple activation_layers.

    batch_size is the number of unique contexts per forward pass through the
    (partial) model — collection is much cheaper than training so this can be
    larger than the training batch size.

    Memory-bounded streaming: when `cache_dir` is provided, work is sorted by
    `(cache_group, chunk_idx, len)` so each chunk's entries arrive contiguously.
    A chunk is flushed once we move past it — exactly one disk read (if prior
    cache exists) and one disk write per chunk, no read-modify-write amplification.
    Caller re-hydrates via `load_activation_cache` (mmap-backed) afterwards.
    Examples without a `cache_group` stay in memory. `SharedContext.context_ids`
    is freed at the end (covers cache-hit contexts too).
    """
    # Group examples by shared context, skipping those already loaded from cache.
    ctx_to_examples: dict[int, list[OracleExample]] = defaultdict(list)
    for ex in examples:
        if ex.cached_activations is not None:
            continue  # already loaded from cache
        ctx_to_examples[id(ex.context)].append(ex)

    # Build work list: one entry per unique context with all layers it needs.
    work: list[
        tuple[SharedContext, list[int], list[tuple[OracleExample, int, list[int]]]]
    ] = []
    for ctx_id, exs in ctx_to_examples.items():
        ctx = exs[0].context
        layers_needed = sorted({ex.activation_layer for ex in exs})
        items = [(ex, ex.activation_layer, ex.context_positions) for ex in exs]
        work.append((ctx, layers_needed, items))

    # Sort by (cache_group, chunk_idx, context_length).
    # Primary key keeps each chunk's items contiguous → one flush per chunk.
    # Secondary length key still gives good intra-chunk padding efficiency.
    def work_sort_key(w):
        ctx, _, items = w
        ex0 = items[0][0]
        return (ex0.cache_group, ex0.original_index // 1000, len(ctx.context_ids))

    work.sort(key=work_sort_key)

    n_to_compute = sum(len(items) for _, _, items in work)
    n_cached = len(examples) - n_to_compute
    n_unique = len(work)
    desc = f"Precompute activations{f' ({label})' if label else ''}"
    if n_cached > 0:
        desc += f" ({n_cached} cached)"
    if n_unique < n_to_compute:
        desc += f" ({n_unique} unique contexts)"
    if n_to_compute == 0:
        print(f"{desc}: all {len(examples)} examples cached, skipping")
        return
    bar = tqdm(total=n_to_compute, desc=desc, unit="ex")

    # Per-chunk pending. When the active chunk_key changes, flush the previous chunk.
    current_key: tuple[str, int] | None = None
    pending_tensors: dict[str, torch.Tensor] = {}
    pending_examples: list[OracleExample] = []

    def flush_current():
        nonlocal current_key
        if cache_dir is None or not pending_tensors:
            current_key = None
            return
        # Wait for queued non-blocking D2H copies before serializing.
        torch.cuda.synchronize()
        _write_chunk(cache_dir, current_key[0], current_key[1], pending_tensors)
        for ex in pending_examples:
            ex._from_cache = True
            ex.cached_activations = None
        pending_tensors.clear()
        pending_examples.clear()
        current_key = None

    for batch_start in range(0, len(work), batch_size):
        batch_work = work[batch_start : batch_start + batch_size]
        B = len(batch_work)

        max_ctx_len = max(len(ctx.context_ids) for ctx, _, _ in batch_work)
        all_layers = sorted({layer for _, layers, _ in batch_work for layer in layers})

        padded = torch.full(
            (B, max_ctx_len), pad_token_id, dtype=torch.long, device=device
        )
        ctx_mask = torch.zeros(B, max_ctx_len, dtype=torch.bool, device=device)
        offsets = []
        for j, (ctx, _, _) in enumerate(batch_work):
            offset = max_ctx_len - len(ctx.context_ids)
            offsets.append(offset)
            padded[j, offset:] = torch.from_numpy(ctx.context_ids).long()
            ctx_mask[j, offset:] = True

        hidden_by_layer = model.collect_activations_multi(
            padded, all_layers, attention_mask=ctx_mask
        )

        # Distribute: extract positions on GPU (tiny indexing), non-blocking D2H copy.
        # When chunk_key changes, flush the previous chunk (handles the boundary
        # batch where two chunks straddle a single batch).
        for j, (ctx, _, items) in enumerate(batch_work):
            offset = offsets[j]
            for ex, layer, positions in items:
                adjusted = [p + offset for p in positions]
                t = hidden_by_layer[layer][j, adjusted].to("cpu", non_blocking=True)
                ex.cached_activations = t
                if cache_dir is not None and ex.cache_group:
                    key = (ex.cache_group, ex.original_index // 1000)
                    if current_key is not None and key != current_key:
                        flush_current()
                    current_key = key
                    pending_tensors[f"{ex.original_index}_{ex.activation_layer}"] = t
                    pending_examples.append(ex)
            bar.update(len(items))

        del padded, ctx_mask, hidden_by_layer
        batch_idx = batch_start // batch_size
        if batch_idx % 100 == 99:
            torch.cuda.empty_cache()

    flush_current()

    # Free raw token buffers on every context — including cache-hit contexts that
    # never entered the work loop. Activations are now cached; context_ids unused.
    for ex in examples:
        ex.context.context_ids = None

    bar.close()


def length_grouped_reorder(
    examples: list[OracleExample],
    batch_size: int,
    window_mult: int = 20,
) -> list[OracleExample]:
    """Reorder examples so similar-length sequences batch together.

    Splits into windows of batch_size * window_mult, sorts each window
    by input_ids length. Reduces padding waste and VRAM variance across batches.
    Matches AO reference length_grouped_reorder (sft.py).
    """
    window = batch_size * window_mult
    result = []
    for i in range(0, len(examples), window):
        chunk = examples[i : i + window]
        chunk.sort(key=lambda ex: len(ex.input_ids))
        result.extend(chunk)
    return result
