"""Transparent JSON loader that handles optional zstd compression.

The result JSONs (Taboo / PersonaQA `details` arrays with per-item chat text)
compress ~10x with zstd. We commit the `.zst` blobs and let the plotting
scripts load either form. If `run_evals.py` writes a fresh plain `.json`,
that takes precedence over a stale `.zst` sibling.
"""

import json
from pathlib import Path


def load_json(path):
    """Load JSON from `path` or `path.zst`, preferring the plain file.

    Accepts a str or Path pointing to either the plain `.json` name or the
    compressed `.zst` name — returns the parsed object either way.
    """
    p = Path(path)
    if p.suffix == ".zst":
        return _read_zst(p)
    if p.exists():
        return json.loads(p.read_text())
    zst = p.with_suffix(p.suffix + ".zst")
    if zst.exists():
        return _read_zst(zst)
    raise FileNotFoundError(f"neither {p} nor {zst} exists")


def _read_zst(path: Path):
    import zstandard

    with open(path, "rb") as f:
        return json.loads(zstandard.decompress(f.read()))
