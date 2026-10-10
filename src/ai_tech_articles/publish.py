#!/usr/bin/env python3
"""
Publish the dataset repo to HuggingFace.

Mirrors the git-tracked tree (minus `.github/`) plus `data/*.parquet` to
datasets/siavava/ai-tech-articles, deleting files there that no longer exist
here, so the HuggingFace repo matches GitHub plus the data.

Usage:
  ai-tech-articles-publish [--repo-id siavava/ai-tech-articles] [--message "..."]
                           [--only-data] [--dry-run]
  (or: python -m ai_tech_articles.publish …)

Token: HF_TOKEN from the environment or from a git-ignored `.env` at the repo
root, else the cached `hf auth login` token.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from ai_tech_articles import REPO_ROOT, load_dotenv

HERE = REPO_ROOT
DATA_PATTERNS = ["README.md", "data/*.parquet"]
#: Never mirrored, even if tracked.
SKIP_PREFIXES = (".github/", ".claude/")
#: The Hub honours a repo's .gitignore at upload time and silently drops
#: matching files; ours ignores data/*.parquet for GitHub's sake, so it must
#: never reach the Hub.
SKIP_FILES = {".gitignore"}


def tracked_files() -> list[str]:
  """Paths git tracks or would track (respecting .gitignore), repo-relative."""
  out = subprocess.run(
    ["git", "-C", str(HERE), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
    check=True,
    capture_output=True,
  ).stdout
  files = [p for p in out.decode().split("\0") if p]
  return sorted(
    f for f in files if not f.startswith(SKIP_PREFIXES) and f not in SKIP_FILES and (HERE / f).is_file()
  )


def mirror_patterns() -> list[str]:
  """Exact paths to upload: the git tree plus the parquet shard(s)."""
  files = tracked_files()
  files += [p.relative_to(HERE).as_posix() for p in sorted((HERE / "data").glob("*.parquet"))]
  return sorted(set(files))


def default_message() -> str:
  """`(update): refresh dataset (N rows)` from the local parquet."""
  try:
    import pyarrow.parquet as pq

    rows = sum(pq.read_metadata(p).num_rows for p in (HERE / "data").glob("*.parquet"))
    return f"(update): refresh dataset ({rows:,} rows)"
  except Exception:  # noqa: BLE001
    return "(update): refresh dataset"


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--repo-id", default="siavava/ai-tech-articles")
  ap.add_argument(
    "--message",
    default=None,
    help='commit subject; defaults to "(update): refresh dataset (N rows)". '
    "Use the repo convention: (feat|fix|update|cleanup): <imperative description>",
  )
  ap.add_argument("--only-data", action="store_true", help="upload just README.md and data/*.parquet")
  ap.add_argument("--dry-run", action="store_true", help="list files that would be uploaded and exit")
  args = ap.parse_args()

  load_dotenv()
  try:
    from huggingface_hub import HfApi
  except ImportError:
    sys.exit("error: huggingface_hub is not installed (uv sync / pip install -e .)")

  if args.only_data:
    allow = DATA_PATTERNS
    delete = ["data/*.parquet"]
  else:
    allow = mirror_patterns()
    delete = ["*"]
  files = sorted(p.relative_to(HERE).as_posix() for pat in allow for p in HERE.glob(pat) if p.is_file())
  if not files:
    sys.exit("error: nothing to upload (run ai-tech-articles-build first)")
  message = args.message or default_message()
  if not re.match(r"^\((feat|fix|update|cleanup)\): \S", message):
    sys.exit(f"error: commit message {message!r} does not follow '(type): description'")

  print(f"{'would upload' if args.dry_run else 'uploading'} to datasets/{args.repo_id}:")
  for f in files:
    size = (HERE / f).stat().st_size
    print(f"  {f} ({size / 1e6:.1f} MB)" if size >= 100_000 else f"  {f}")
  if args.dry_run:
    return

  api = HfApi(token=os.environ.get("HF_TOKEN"))
  commit = api.upload_folder(
    repo_id=args.repo_id,
    repo_type="dataset",
    folder_path=str(HERE),
    allow_patterns=allow,
    delete_patterns=delete,
    commit_message=message,
  )
  print(f"done: {commit}")
  verify_parquet(api, args.repo_id, files)


def verify_parquet(api, repo_id: str, uploaded: list[str]) -> None:
  """Fail loudly if a parquet on the Hub does not match the local bytes.

  Guards against the Hub ignoring an upload (for example because a .gitignore
  rule in the repo matched it), which otherwise leaves a commit with no data.
  """
  import hashlib

  remote = {
    e.path: (getattr(e, "lfs", None) or {}).get("sha256")
    for e in api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo="data", expand=True)
  }
  for f in uploaded:
    if not f.endswith(".parquet"):
      continue
    local_sha = hashlib.sha256((HERE / f).read_bytes()).hexdigest()
    if remote.get(f) != local_sha:
      sys.exit(
        f"error: {f} on the Hub does not match the local file (hub {remote.get(f)}, local {local_sha})"
      )
  print("verified: parquet on the Hub matches local")


if __name__ == "__main__":
  main()
