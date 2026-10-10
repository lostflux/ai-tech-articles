#!/usr/bin/env python3
"""
Publish the dataset: one git history, two remotes.

Commits `data/*.parquet` (LFS pointers; see .gitattributes) and README.md,
then pushes the branch, LFS objects included, to the HuggingFace remote (`hf`)
and to GitHub (`origin`). Both remotes show identical history. GitHub's LFS
quota (1 GB free) is what the incremental shards protect: a daily publish adds
one small shard; only `build --full` re-uploads everything.

Usage:
  ai-tech-articles-publish [--message "(update): …"] [--as-bot] [--dry-run]
  (or: python -m ai_tech_articles.publish …)

Token: HF_TOKEN from the environment or from a git-ignored `.env` at the repo
root. The `hf` remote is added if missing.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys

from ai_tech_articles import HF_REPO_ID, REPO_ROOT, load_dotenv

HF_REMOTE = "hf"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"
BOT = ["-c", "user.name=evilalt", "-c", "user.email=amittaijoel@gmail.com"]
MESSAGE_RE = re.compile(r"^\((feat|fix|update|cleanup)\): \S")


def git(*args: str, check: bool = True, capture: bool = True, extra: list[str] | None = None) -> str:
  cmd = ["git", "-C", str(REPO_ROOT), *(extra or []), *args]
  res = subprocess.run(cmd, check=check, capture_output=capture, text=True)
  return (res.stdout or "").strip()


def default_message() -> str:
  """`(update): refresh dataset (N rows)` from the local parquet."""
  try:
    import pyarrow.parquet as pq

    rows = sum(pq.read_metadata(p).num_rows for p in (REPO_ROOT / "data").glob("*.parquet"))
    return f"(update): refresh dataset ({rows:,} rows)"
  except Exception:  # noqa: BLE001
    return "(update): refresh dataset"


def hf_credential_env() -> dict[str, str]:
  """Environment that makes `git push hf` authenticate with HF_TOKEN.

  The token never appears on a command line: git asks the helper, which reads
  it from the environment.
  """
  token = os.environ.get("HF_TOKEN")
  if not token:
    sys.exit("error: HF_TOKEN is not set (environment or .env)")
  env = dict(os.environ)
  env["HF_TOKEN"] = token
  env["GIT_CONFIG_COUNT"] = "1"
  env["GIT_CONFIG_KEY_0"] = "credential.helper"
  env["GIT_CONFIG_VALUE_0"] = '!f() { echo username=hf_user; echo "password=$HF_TOKEN"; }; f'
  return env


def ensure_hf_remote() -> None:
  remotes = git("remote").split()
  if HF_REMOTE not in remotes:
    git("remote", "add", HF_REMOTE, HF_URL)
  elif git("remote", "get-url", HF_REMOTE) != HF_URL:
    git("remote", "set-url", HF_REMOTE, HF_URL)


def verify_parquet() -> None:
  """Fail loudly if the parquet on the Hub does not match the local bytes."""
  from huggingface_hub import HfApi

  api = HfApi(token=os.environ.get("HF_TOKEN"))
  remote = {
    e.path: (getattr(e, "lfs", None) or {}).get("sha256")
    for e in api.list_repo_tree(HF_REPO_ID, repo_type="dataset", path_in_repo="data", expand=True)
  }
  for p in sorted((REPO_ROOT / "data").glob("*.parquet")):
    rel = p.relative_to(REPO_ROOT).as_posix()
    local_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    if remote.get(rel) != local_sha:
      sys.exit(
        f"error: {rel} on the Hub does not match the local file (hub {remote.get(rel)}, local {local_sha})"
      )
  print("verified: parquet on the Hub matches local")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument(
    "--message", default=None, help='commit subject; default "(update): refresh dataset (N rows)"'
  )
  ap.add_argument("--as-bot", action="store_true", help="commit as the evilalt utility identity, unsigned")
  ap.add_argument("--dry-run", action="store_true", help="show what would be committed and pushed")
  args = ap.parse_args()
  load_dotenv()

  branch = git("rev-parse", "--abbrev-ref", "HEAD")
  if branch != "main":
    sys.exit(f"error: publish from main, not {branch}")
  message = args.message or default_message()
  if not MESSAGE_RE.match(message):
    sys.exit(f"error: commit message {message!r} does not follow '(type): description'")

  git("add", "--", "data", "README.md")
  staged = git("diff", "--cached", "--name-only")
  other = [
    line
    for line in git("status", "--porcelain").splitlines()
    if line[:2].strip() and line[3:] not in staged.split()
  ]
  if other:
    sys.exit("error: unrelated uncommitted changes; commit or stash them first:\n  " + "\n  ".join(other))

  print(f"branch {branch}; staged: {staged.split() or 'nothing'}")
  if staged:
    print(f"commit: {message}")
  ahead = (
    git("rev-list", "--count", f"{HF_REMOTE}/main..HEAD", check=False)
    if HF_REMOTE in git("remote").split()
    else "?"
  )
  print(f"push: {HF_REMOTE} and origin (LFS objects included); {ahead} commit(s) not yet on {HF_REMOTE}")
  if args.dry_run:
    git("reset", "-q", "--", "data", "README.md")
    return

  if staged:
    if args.as_bot:
      git("commit", "-q", "--no-gpg-sign", "-m", message, extra=BOT)
    else:
      git("commit", "-q", "-m", message)
  ensure_hf_remote()
  git("lfs", "install", "--local")
  git("fetch", "-q", "origin", "main")
  if git("rev-list", "--count", "HEAD..origin/main") != "0":
    git("rebase", "-q", "origin/main")
  env = hf_credential_env()
  subprocess.run(["git", "-C", str(REPO_ROOT), "push", HF_REMOTE, "main"], check=True, env=env)
  subprocess.run(["git", "-C", str(REPO_ROOT), "push", "origin", "main"], check=True)
  print(f"pushed: {git('rev-parse', '--short', 'HEAD')} to {HF_REMOTE} and origin")
  verify_parquet()


if __name__ == "__main__":
  main()
