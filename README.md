---
language:
- en
license: mit
task_categories:
- text-generation
- feature-extraction
pretty_name: AI/Technology Articles
tags:
- temporal series data
- language data
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: year
    dtype: int64
  - name: title
    dtype: string
  - name: url
    dtype: string
  - name: text
    dtype: string
  - name: date
    dtype: string
  splits:
  - name: train
    num_bytes: 570306748
    num_examples: 63931
  download_size: 200736459
  dataset_size: 570306748
---

# AI/Tech Dataset

This dataset is a collection of AI/tech articles scraped from the web: the 2023
corpus collected by the original Haskell scraper plus everything the Rust
crawler has added since (2023 onwards, with full publication dates).

It's hosted on [HuggingFace Datasets](https://huggingface.co/datasets/siavava/ai-tech-articles), so it is easier to load in and work with.

## To load the dataset

### 1. Install [HuggingFace Datasets](https://huggingface.co/docs/datasets/installation.html)

```bash
pip install datasets
```

### 2. Load the dataset

```python
from datasets import load_dataset

dataset = load_dataset("siavava/ai-tech-articles")

# optionally, convert it to a pandas dataframe:
df = dataset["train"].to_pandas()
```

You do not need to clone this repo.
HuggingFace will download the dataset for you, the first time that you load it,
and cache it locally so it does not need to re-download it again
(unless it detects a change upstream).

## Columns

| column | type | notes |
| --- | --- | --- |
| `id` | int64 | row number: 0-based and sequential, in collection order (the 2023 rows first). A new crawl only appends, so existing ids keep their place; removing earlier rows (a stricter relevance rule) renumbers. `url` is the stable key across releases |
| `year` | int64 | publication year |
| `title` | string | article title |
| `url` | string | article url |
| `text` | string | article text, one paragraph per line |
| `date` | string | publication date (`YYYY-MM-DD`); set for everything collected since 2023, null for the original 2023 collection |

## Notebooks

- [`example.ipynb`](example.ipynb): load the dataset and look at a few rows.
- [`analytics.ipynb`](analytics.ipynb): counts by year, month and
  domain, article lengths, sample titles.

## Building the dataset (maintainers)

This repo is a small Python project (`src/ai_tech_articles`). It turns the raw
output of the [scraper](https://github.com/lostflux/functional-scraper) into the
parquet that HuggingFace serves and publishes it. The scraper repo is expected
as a sibling directory (`../functional-scraper`); pass `--scraper-dir` otherwise.

```bash
uv sync                                        # or: pip install -e .
uv run ai-tech-articles-build --update-readme  # append a shard with the rows not yet published
uv run ai-tech-articles-build --full --update-readme  # rebuild everything into one shard
uv run ai-tech-articles-publish --dry-run      # what would be committed and pushed
uv run ai-tech-articles-publish                # commit the parquet + card, push to both remotes
```

`build` filters by year (2000 to now), drops empty texts, deduplicates by
normalised URL, applies the scraper's strong-term relevance rule
(`strong_targets` in the scraper's `config.yml`) to rows collected since 2023
(`--no-relevance-filter` keeps everything crawled), and numbers rows
sequentially in collection order. `--update-readme` refreshes the counts in
this file's front matter.

One git history, two remotes. `publish` commits `data/*.parquet` (Git LFS
pointers, per `.gitattributes`) and `README.md`, then pushes the same commit,
LFS objects included, to the HuggingFace remote `hf` and to GitHub `origin`.
The dataset is a set of immutable shards (`data/train-NNNNN.parquet`): a
normal `build` appends one shard with only the rows not yet published, ids
continuing from the last shard, so a daily publish costs a few megabytes on
each remote. `build --full` rebuilds everything into one shard for schema or
rule changes; it re-uploads the whole dataset, which matters for GitHub's
1 GB free LFS quota. Commit messages follow `(type): description`; `publish`
refuses anything else. Inside Python, `ai_tech_articles.load_dataframe()`
returns the dataset from the local shards when present, else from HuggingFace.

### Automation

`.github/workflows/build-dataset.yml` runs build and publish on a runner:
triggered by the scraper repo's daily crawl (`repository_dispatch`, event
`crawl-finished`), on demand, and weekly as a safety net. Secrets in this
repo: `HF_TOKEN` (write access to the dataset) and optionally `ACTOR_TOKEN`
(PAT of the `evilalt` utility account) so the push to GitHub is made by that
account. Locally, put the token in a git-ignored `.env` (`HF_TOKEN=hf_…`).

- `data/*.parquet`: compressed [parquet](https://www.databricks.com/glossary/what-is-parquet) containing the data.
- For the raw text files, see the [scraper repo](https://github.com/lostflux/functional-scraper) on GitHub.
