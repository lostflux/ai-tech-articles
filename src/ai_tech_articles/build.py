#!/usr/bin/env python3
"""
Build the dataset parquet from the scraper's raw output.

Sources:
  A. legacy (Haskell scraper, 2023): <scraper>/data/log/N and N.txt pairs
  B. new (Rust scraper):             <scraper>/data/articles/*.jsonl.zst

Usage:
  ai-tech-articles-build [--scraper-dir ../functional-scraper]
                         [--out data/train-00000-of-00001.parquet] [--update-readme]
  (or: python -m ai_tech_articles.build …)
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from ai_tech_articles import REPO_ROOT, SCHEMA

HERE = REPO_ROOT


def domain_of(url: str) -> str:
  host = (urlsplit(url).hostname or "").lower()
  return host[4:] if host.startswith("www.") else host


def normalize_url(url: str) -> str:
  parts = urlsplit(url.strip())
  host = (parts.netloc or "").lower()
  if host.startswith("www."):
    host = host[4:]
  path = parts.path.rstrip("/")
  return f"{parts.scheme.lower()}://{host}{path}"


def parse_year(value) -> int | None:
  if value is None:
    return None
  try:
    return int(str(value).strip())
  except ValueError:
    return None


def read_legacy(log_dir: Path):
  """Yield legacy records in id order."""
  if not log_dir.is_dir():
    print(f"warning: legacy dir {log_dir} not found, skipping", file=sys.stderr)
    return
  ids = sorted(int(p.name) for p in log_dir.iterdir() if p.name.isdigit())
  for n in ids:
    meta_path = log_dir / str(n)
    text_path = log_dir / f"{n}.txt"
    if not text_path.exists():
      yield {"id": n, "_missing_text": True}
      continue
    with open(meta_path, encoding="utf-8", errors="replace") as fh:
      lines = iter(fh.read().split("\n"))
    title = next(lines, "").strip()
    year = next(lines, "").strip()
    if year.startswith("|"):
      title = f"{title} {year}"
      year = next(lines, "").strip()
    url = next(lines, "").strip()
    with open(text_path, encoding="utf-8", errors="replace") as fh:
      text = fh.read().rstrip()
    yield {
      "id": n,
      "year": parse_year(year),
      "title": title,
      "url": url,
      "text": text,
      "date": None,
      "domain": domain_of(url),
      "source": "haskell-2023",
      "fetched_at": None,
    }


def read_new(articles_dir: Path):
  """Yield new-scraper records, shards in filename order, ids sorted within the source."""
  if not articles_dir.is_dir():
    return
  shards = sorted(articles_dir.glob("*.jsonl.zst"))
  if not shards:
    return
  import zstandard

  records = []
  for shard in shards:
    with open(shard, "rb") as raw:
      reader = zstandard.ZstdDecompressor().stream_reader(raw)
      for line in io.TextIOWrapper(reader, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
          continue
        obj = json.loads(line)
        url = obj.get("url") or ""
        records.append(
          {
            "id": int(obj["id"]),
            "year": parse_year(obj.get("year")),
            "title": (obj.get("title") or "").strip(),
            "url": url,
            "text": (obj.get("text") or "").rstrip(),
            "date": obj.get("date"),
            "domain": obj.get("domain") or domain_of(url),
            "source": "rust",
            "fetched_at": obj.get("fetched_at"),
          }
        )
  records.sort(key=lambda r: r["id"])
  yield from records


def load_relevance(scraper_dir: Path):
  """The crawler's strong-term rule from config.yml, or None if unset."""
  try:
    import yaml

    cfg = yaml.safe_load((scraper_dir / "config.yml").read_text(encoding="utf-8")) or {}
  except Exception as e:  # noqa: BLE001
    print(f"warning: could not read config.yml ({e}); relevance filter off", file=sys.stderr)
    return None
  strong = [str(t).strip().lower() for t in cfg.get("strong_targets") or [] if str(t).strip()]
  if not strong:
    return None
  crawl = cfg.get("crawl") or {}
  return {
    "single": {t for t in strong if " " not in t},
    "phrases": [t for t in strong if " " in t],
    "min_mentions": int(crawl.get("min_strong_mentions", 3)),
    "require_lede": bool(crawl.get("require_strong_in_lede", True)),
    "lede_chars": int(crawl.get("lede_chars", 600)),
    "dense_mentions": int(crawl.get("dense_strong_mentions", 12)),
    "dense_per_1000": float(crawl.get("dense_strong_per_1000", 5.0)),
    "dense_distinct": int(crawl.get("dense_strong_distinct", 2)),
  }


def _tokens(text: str) -> list[str]:
  return [t for t in re.split(r"[^0-9a-z]+", text.lower().replace("a.i.", " ai ")) if t]


def _scan(rule: dict, text: str):
  toks = _tokens(text)
  stream = " " + " ".join(toks) + " "
  found, mentions = set(), 0
  for t in toks:
    if t in rule["single"]:
      found.add(t)
      mentions += 1
  for p in rule["phrases"]:
    c = stream.count(" " + p + " ")
    if c:
      found.add(p)
      mentions += c
  return found, mentions, len(toks)


def is_relevant(rule: dict, title: str, text: str) -> bool:
  """Mirror of the Rust classifier's strong-term rule (scraper-extract)."""
  found, mentions, n = _scan(rule, text)
  tfound, tmentions, _ = _scan(rule, title)
  mentions += tmentions
  if mentions < rule["min_mentions"]:
    return False
  if not rule["require_lede"]:
    return True
  lfound, _, _ = _scan(rule, text[: rule["lede_chars"]])
  dense = (
    mentions >= rule["dense_mentions"]
    and 1000 * mentions / max(n, 1) >= rule["dense_per_1000"]
    and len(found) >= rule["dense_distinct"]
  )
  return bool(tfound) or bool(lfound) or dense


def build(scraper_dir: Path, relevance: dict | None = None):
  current_year = dt.date.today().year
  dropped = Counter()
  seen_urls: set[str] = set()
  seen_ids: set[int] = set()
  rows = []

  sources = [
    read_legacy(scraper_dir / "data" / "log"),
    read_new(scraper_dir / "data" / "articles"),
  ]
  for source in sources:
    for rec in source:
      if rec.get("_missing_text"):
        dropped["missing text file"] += 1
        continue
      y = rec["year"]
      if y is None:
        dropped["missing year"] += 1
        continue
      if not (2000 <= y <= current_year):
        dropped["year out of range"] += 1
        continue
      if not rec["text"].strip():
        dropped["empty text"] += 1
        continue
      if (
        relevance is not None
        and rec["source"] == "rust"
        and not is_relevant(relevance, rec["title"], rec["text"])
      ):
        dropped["not about AI (strong-term rule)"] += 1
        continue
      key = normalize_url(rec["url"])
      if key in seen_urls:
        dropped[f"duplicate url ({rec['source']})"] += 1
        continue
      if rec["id"] in seen_ids:
        dropped[f"duplicate id ({rec['source']})"] += 1
        continue
      seen_urls.add(key)
      seen_ids.add(rec["id"])
      rows.append(rec)

  # Number rows in storage order so a later crawl only appends and existing ids keep their place.
  rows.sort(key=lambda r: r["id"])
  for i, r in enumerate(rows):
    r["id"] = i
  by_source = Counter(r["source"] for r in rows)
  columns = {name: [r[name] for r in rows] for name in SCHEMA.names}
  table = pa.Table.from_pydict(columns, schema=SCHEMA)
  return table, dropped, by_source


def update_readme(readme: Path, table: pa.Table, download_size: int) -> None:
  content = readme.read_text(encoding="utf-8")
  m = re.match(r"^---\n(.*?)\n---\n", content, re.S)
  if not m:
    raise SystemExit(f"{readme}: no YAML front matter found")
  front, body = m.group(1), content[m.end() :]

  features = [{"name": f.name, "dtype": "int64" if pa.types.is_integer(f.type) else "string"} for f in SCHEMA]
  num_bytes = int(table.nbytes)

  try:
    import yaml
  except ImportError:
    yaml = None

  if yaml is not None:
    meta = yaml.safe_load(front) or {}
    info = meta.get("dataset_info") or {}
    splits = info.get("splits") or [{"name": "train"}]
    splits[0]["num_bytes"] = num_bytes
    splits[0]["num_examples"] = table.num_rows
    new_info = dict(info)
    new_info["features"] = features
    new_info["splits"] = splits
    new_info["download_size"] = download_size
    new_info["dataset_size"] = num_bytes
    meta["dataset_info"] = new_info
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000).rstrip("\n")
  else:
    lines = ["dataset_info:", "  features:"]
    for f in features:
      lines += [f"  - name: {f['name']}", f"    dtype: {f['dtype']}"]
    lines += [
      "  splits:",
      "  - name: train",
      f"    num_bytes: {num_bytes}",
      f"    num_examples: {table.num_rows}",
      f"  download_size: {download_size}",
      f"  dataset_size: {num_bytes}",
    ]
    block = "\n".join(lines)
    pattern = re.compile(r"^dataset_info:\n(?:[ -].*\n?)*", re.M)
    if pattern.search(front + "\n"):
      front = pattern.sub(block + "\n", front + "\n", count=1).rstrip("\n")
    else:
      front = front + "\n" + block

  readme.write_text(f"---\n{front}\n---\n{body}", encoding="utf-8")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--scraper-dir", type=Path, default=HERE.parent / "functional-scraper")
  ap.add_argument("--out", type=Path, default=Path("data/train-00000-of-00001.parquet"))
  ap.add_argument("--update-readme", action="store_true", help="rewrite README.md front matter stats")
  ap.add_argument(
    "--no-relevance-filter",
    action="store_true",
    help="keep every stored record; by default new-scraper rows must pass the strong-term rule in config.yml",
  )
  args = ap.parse_args()

  out = args.out if args.out.is_absolute() else HERE / args.out
  scraper_dir = args.scraper_dir.resolve()

  relevance = None if args.no_relevance_filter else load_relevance(scraper_dir)
  table, dropped, by_source = build(scraper_dir, relevance)

  out.parent.mkdir(parents=True, exist_ok=True)
  for stale in out.parent.glob("train-*.parquet"):
    if stale.resolve() != out.resolve():
      print(f"removing stale shard {stale.name}")
      stale.unlink()
  pq.write_table(table, out, compression="zstd")
  size = out.stat().st_size

  print(f"\nrows: {table.num_rows}")
  print("by source: " + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
  years = Counter(table["year"].to_pylist())
  print("by year:   " + " ".join(f"{y}:{c}" for y, c in sorted(years.items())))
  print("dropped:   " + (", ".join(f"{k}={v}" for k, v in sorted(dropped.items())) or "none"))
  print(f"wrote {out} ({size / 1e6:.1f} MB, in-memory {table.nbytes / 1e6:.1f} MB)")

  if args.update_readme:
    update_readme(HERE / "README.md", table, size)
    print("updated README.md front matter")


if __name__ == "__main__":
  main()
