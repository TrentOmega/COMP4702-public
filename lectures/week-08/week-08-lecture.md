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

# Week 8 Lecture Notes: Neural Networks and Deep Learning

## Contents

- [Scope for Week 8](#scope-for-week-8)
- [Learning goals for this notebook](#learning-goals-for-this-notebook)
- [Week 8 setup status](#week-8-setup-status)
- [Core sections to complete](#core-sections-to-complete)
- [Past exam anchors already identified](#past-exam-anchors-already-identified)
- [Sources currently available](#sources-currently-available)

## Scope for Week 8

- Topic: neural networks and deep learning.
- Exam-index topic label for Week 8: neural networks and deep learning (CNNs).
- Default focus for follow-up notes: multilayer perceptrons, backpropagation, convolution, pooling, and output-layer interpretation.
- Practical alignment: pending the official Week 8 prac brief being added to the repo.

## Learning goals for this notebook

1. Explain what a neural network is trying to represent and why hidden layers increase expressiveness.
2. Connect forward propagation, loss computation, and backpropagation into one trainable pipeline.
3. Distinguish dense feed-forward networks from convolutional neural networks and identify where CNN-specific operations matter.

```python
import random
import numpy as np
import matplotlib.pyplot as plt

SEED = 4702
random.seed(SEED)
np.random.seed(SEED)
print(f"Seed set to {SEED}")
```

## Week 8 setup status

This is a baseline scaffold for Week 8.

- The repo already indicates that Week 8 is the neural-network / deep-learning week via the exam index.
- The course summary PDF is present locally, but the usual PDF extraction tools are unavailable in this shell session, so this scaffold avoids inventing reading-specific details.
- The notebook is ready to be expanded once the exact Week 8 reading and any lecture transcript are available.

## Core sections to complete

### 1. Neural network objective

- Define the supervised objective for binary and multiclass settings.
- State what parameters are being learned.
- Connect logits, activations, and loss functions.

### 2. Forward propagation

- Layer notation.
- Affine transform plus non-linearity.
- Output-layer choices for regression, binary classification, and multiclass classification.

### 3. Backpropagation

- Chain rule structure.
- Gradient flow from loss to earlier layers.
- Why activation choice affects optimisation.

### 4. Training dynamics

- Learning rate.
- Overfitting and regularisation.
- Mini-batch vs full-batch vs stochastic updates.

### 5. CNN-specific ideas

- Convolution as shared-weight feature extraction.
- Stride and pooling.
- Why CNNs reduce parameter count relative to fully connected image models.

## Past exam anchors already identified

- `2023-B-6`: convolution on a small matrix by hand.
- `2024-A-7`: CNN concepts including stride and pooling.
- `2025-B-8`: feed-forward network setup for bitmap-style inputs, with Week 7 overlap.

## Sources currently available

- [Week-by-week exam index](../../references/exam_questions_2023_2025_by_week.csv)
- [Extracted 2023-2025 exams](../../references/COMP4702_exams_2023_2025.md)
- [Course summary table PDF](../../references/CourseSummaryTable_v1_26.pdf)
- [Main textbook (Lindholm)](../../references/main-text-book-machine-learning-lindholm-2022.pdf)
