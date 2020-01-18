#!/usr/bin/env python3

# std
from pathlib import Path

# 3rd

# ours
from clusterking import Data

if __name__ == "__main__":
    this_dir = Path(".").parent
    files = this_dir.iterdir()
    for f in files:
        if not f.name.endswith(".sql"):
            continue
        print(f.name)
        data = Data(f)
        data.df.to_csv(f.parent / f.name.replace("sql", "csv"))
