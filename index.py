#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  Simple script to generate metadata about corpus.
"""

__author__  = "Amittai Siavava"
__version__ = "0.0.1"

from os import mkdir
from collections import Counter
import csv
import pandas as pd

def index_pages():
  """
    Generate a friendly index of the pages.

    We create a csv and a tsv (in case one proves more convenient than the other).
  """
  docID = 0

  with open("raw2.csv", "w") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["id", "year", "title", "url", "text"])
    while True:
      try:
        with open(f"../data/log/{docID}", "r") as meta, open(f"../data/log/{docID}.txt", "r") as data:
          title = meta.readline().strip()
          year = meta.readline().strip()
          # if year starts with '|', append to title and read next line
          if year.startswith("|"):
            title += " " + year
            year = meta.readline().strip()
          url = meta.readline().strip()
          text = data.read()

          meta.close()
          data.close()

          if year and 2000 <= int(year) <= 2023:
            print(f"Indexing: {docID}")
            writer.writerow([docID, year, f'"{title}"', f'"{url}"', f'"{text}"'])
          docID += 1
      except Exception as e:
        print(e)
        print(f"{docID = }")
        break
        
  print("Done.")
  print(f"okay...")
  # save to file
  
  # df = pd.read_csv("raw.csv")
  print("Done.")

def categorize():
  """
    Categorize the pages by year.
  """

  docID = 0
  years = Counter()
  years_index = { str(year) for year in range(2000, 2024)}
  while True:
    try:
      with open(f"../log/{docID}.txt", "r") as doc, open(f"../log/{docID}", "r") as meta:
        title = meta.readline().strip()
        year = meta.readline().strip()
        url = meta.readline().strip()
        text = doc.read()
        doc.close()
        meta.close()

        if year in years_index:
          try:
            mkdir(f"../categorized/{year}")
          except:
            pass

          id = years[year]
          with open(f"../categorized/{year}/{id}.txt", "w") as f:
            f.write(f"{title}\n{year}\n{url}\n\n{text}")
            f.close()
          years[year] += 1
        docID += 1
        
    except:
      break


if __name__ == "__main__":
  index_pages()
  # categorize()
  # load_data()
