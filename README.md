---
language: 
  - en
pretty_name: "AI/Technology Articles"
tags:
  - temporal series data
  - language data
license: mit
task_categories:
  - text-generation
  - feature-extraction
---

# AI/Tech Dataset

This dataset is a collection of AI/tech articles scraped from the web.

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

> [!NOTE]
> You do not need to clone this repo.
>
> HuggingFace will download the dataset for you, the first time that you load it,
> and cache it so it does not need to re-download it again
> (unless it detects a change upstream).

## File Structure

- [analytics.ipynb](analytics.ipynb) - Notebook containing some details about the dataset and how to load it.
- [data/index.parquet](./index.csv) - compressed [parquet](https://www.databricks.com/glossary/what-is-parquet) containing the data.
- For raw text files, see the [scraper repo](https://github.com/siavava/scrape.hs) on GitHub.
