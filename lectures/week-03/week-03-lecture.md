---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md,py:percent
    notebook_metadata_filter: kernelspec,jupytext
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Notebook

Created by scripts/setup_notebook_workflow.sh

```python
import random
import numpy as np

SEED = 4702
random.seed(SEED)
np.random.seed(SEED)
print(f"Seed set to {SEED}")
```
