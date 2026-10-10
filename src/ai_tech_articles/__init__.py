"""
ai-tech-articles: build, publish and load the AI/tech articles dataset.

* ``ai_tech_articles.build``   — merge the scraper's raw output into parquet
* ``ai_tech_articles.publish`` — upload README + parquet to HuggingFace
* :func:`load_dataframe`       — the dataset as a pandas DataFrame, from the
  local parquet when present, else from HuggingFace
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

__version__ = "0.2.0"

#: Repository root (the package lives in ``src/ai_tech_articles``).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
HF_REPO_ID = "siavava/ai-tech-articles"

SCHEMA = pa.schema(
  [
    ("id", pa.int64()),
    ("year", pa.int64()),
    ("title", pa.string()),
    ("url", pa.string()),
    ("text", pa.string()),
    ("date", pa.string()),
  ]
)


def load_dotenv(path: Path | None = None) -> list[str]:
  """Load ``KEY=VALUE`` lines from ``.env`` at the repo root into ``os.environ``.

  Existing environment variables win. Returns the keys that were set. Kept
  dependency-free on purpose; ``.env`` is git-ignored.
  """
  import os

  path = path or REPO_ROOT / ".env"
  if not path.is_file():
    return []
  loaded = []
  for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    key, value = line.split("=", 1)
    key = key.strip().removeprefix("export ").strip()
    value = value.strip().strip("'\"")
    if key and key not in os.environ:
      os.environ[key] = value
      loaded.append(key)
  return loaded


def local_parquet_files() -> list[Path]:
  """The parquet shard(s) in ``data/``, if the repo is checked out."""
  return sorted(DATA_DIR.glob("train-*.parquet"))


def load_table(local_first: bool = True) -> pa.Table:
  """The dataset as a pyarrow Table.

  Reads ``data/train-*.parquet`` when it exists (a fresh ``build`` may be
  newer than what is published); otherwise downloads from HuggingFace via
  ``datasets`` (``pip install datasets``).
  """
  files = local_parquet_files() if local_first else []
  if files:
    import pyarrow.parquet as pq

    return pa.concat_tables([pq.read_table(f) for f in files])
  from datasets import load_dataset

  return load_dataset(HF_REPO_ID, split="train").data.table


def load_dataframe(local_first: bool = True):
  """:func:`load_table` as a pandas DataFrame."""
  return load_table(local_first).to_pandas()
