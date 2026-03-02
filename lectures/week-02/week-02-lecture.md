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

# Week 2 Lecture Notes: Lindholm Chapter 2 (Exam-Focused)

## Scope for Week 2

- Topic: supervised learning.
- Core methods: `k`-nearest neighbours and decision trees.
- Reading: Lindholm (2022), Chapter 2.
- Prac alignment: Week 2 focuses on `k`-NN and decision trees.

## Learning goals for this notebook

1. Explain Chapter 2 concepts clearly in ML language.
2. Prioritise concepts that repeatedly appear in 2023, 2024, and 2025 exams.
3. Build intuition with small, reproducible Python toy examples.

```python
import random
import numpy as np

SEED = 4702
random.seed(SEED)
np.random.seed(SEED)
print(f"Seed set to {SEED}")
```

## Chapter 2 summary (Lindholm): what matters most

Supervised learning is a catagory of machine learning where all training data has associated output labels. The model is then trained to predict an output from an unknown input value.

### 1) Supervised learning setup

- Training data is labelled pairs: `T = {(x_i, y_i)}_{i=1}^n`.
- Input `x` can be high-dimensional (`p` features).
- Output `y` type defines the problem:
  - regression: `y` numerical,
  - classification: `y` categorical.

### 2) Generalisation is the objective

Chapter 2 emphasises this explicitly: fitting training data is not enough.
The useful model is the one that predicts well on *new* unseen inputs.

### 3) Parametric/non-parametric methods
- **Non-parametric methods** use the training data for making predictions
- **Parmetric methods** use a model governed by a fixed number of parameters.

### 3) k-NN classifier (distance-based non-parametric method)

- Predict using nearby training points in input space.
- Classification: majority vote among the `k` nearest neighbours.
- Regression: average of neighbour targets.
- `k` is a **hyperparameter** (chosen by user, not learned directly).
- A **decision boundary** for a classifyer is the point in the input space where the class prediction abruptly changes.

Key tradeoff:
- small `k` (especially `k=1`) -> very flexible, often overfits,
- large `k` -> smoother, can underfit.

### 5) Input normalisation
- Input data needs to be normalised otherwise certain input dimensions will dominate the results
- Two methods
  - [codex complete from lindholm]
  - [codex complete from lindholm]

### 4) Decision trees (rule-based method)

- Split input space with binary rules (`x_j < s`).
- Leaves output constant predictions:
  - class vote (classification),
  - mean value (regression).
- Definitions:
  - **leaf nodes** are the end points of the branching which are different regions of the input space.
  - **internal nodes** are the internal splits of the tree.
  - **branches** are the lines connecting the nodes.
- Searching through all possible binary trees is not possible for any practicle case (the problem is NP-complete)
- **Recursive binary splitting** is the heuristic algorithm for learning the tree.
  - Greedy algorithm: [codex give a short one sentence explaination]
  - **Learning a regression tree**
    - Each leaf node will have a predicted out put y_hat_l = av{y_i : **x_i** [belongs to] R_l}, where l is leaf node 1,2,...,L, i is the data point 1,2,...,n
    - It will search through each 
- Trees are also non-linear and piecewise-constant predictors.

### 5) Split criteria for classification trees

Common impurity/cost choices at candidate split nodes:
- misclassification rate,
- Gini index,
- entropy.

Practical Chapter 2 point: Gini/entropy are often preferred for splitting because they are more sensitive to node purity changes than misclassification rate.

### 6) Typical variable symbols
p, dimension of the input vector, variable j
n, number of data points, variable i
L, number of distinct regions in the decision tree, variable l


## Past exam-oriented priorities (23-25)

These concepts recur directly in MCQs and longer questions:

1. supervised-learning definition and labelled data setup,
2. hold-out/test purpose and generalisation,
3. `k` in `k`-NN as a hyperparameter,
4. overfitting/underfitting as model complexity changes (`k` or tree depth),
5. decision-tree predictions, splits, and misclassification rate,
6. split-quality measures: entropy, Gini, misclassification,
7. confusion matrix for classification performance,
8. logistic decision boundary output `0.5` (bridge concept to Chapter 3, but appears repeatedly in exams).

Fast exam checklist:

- Can I explain why `k=1` can have zero training error yet still generalise poorly?
- Can I compute a decision-tree misclassification rate from leaf outcomes?
- Can I choose valid statements about entropy/Gini/misclassification as split criteria?
- Can I build a confusion matrix from predicted vs true labels?
- Can I state why decision boundary output is `0.5` for standard logistic binary classification?

## Past Exam Questions (23-25)

Note: These are copied from the exam text extraction file: [COMP4702_exams_2023_2025.md](../../references/COMP4702_exams_2023_2025.md).

### 2023 exam

- Question 4. What does the “k” in the kNN algorithm represent? ([2023 exam PDF](../../references/2023_COMP4702_exam.pdf), Part A)
  - (a) The number of training examples used for each prediction.
  - (b) The number of features in the dataset.
  - (c) The number of classes in the dataset.
  - (d) The regularisation term in the loss function.
- Question 5. Which of the following metrics is NOT suitable for evaluating the quality of a split in a decision tree for classification? ([2023 exam PDF](../../references/2023_COMP4702_exam.pdf), Part A)
  - (a) Entropy.
  - (b) Gini index.
  - (c) Misclassification rate.
  - (d) Sensitivity.
- Question 1(a) (Part B): ([2023 exam PDF](../../references/2023_COMP4702_exam.pdf), Part B)
  - (a) Consider using a decision stump (i.e. a decision tree with depth = 1) to classify this data.
    (i) From Figure 1, specify the trained model (approximately) in terms of an IF...THEN decision
    rule.                                                                           (2 marks)
  - (ii) State the misclassification rate of the trained model on the training set and explain
    why.                                                                               (2 marks)

### 2024 exam

- Question 2. In the k-nearest neighbours method (k-NN), k refers to: ([2024 exam PDF](../../references/2024_COMP4702_exam.pdf), Part A)
  - (a) The maximum number of training iterations.
  - (b) The number of data points that are used to determine any prediction made by the model.
  - (c) The learning rate parameter.
  - (d) The number of base models that are combined to produce the final output.
- Question 1(a) (Part B): ([2024 exam PDF](../../references/2024_COMP4702_exam.pdf), Part B)
  - (a) A decision tree is trained on this data and creates the discriminant function/decision boundary
    shown by the shaded rectangle, C. Draw a diagram of this decision tree, including the rule/decision
    made at each node and the final classification at each leaf node.                        (4 marks)
- Question 1(c) (Part B): ([2024 exam PDF](../../references/2024_COMP4702_exam.pdf), Part B)
  - (c) Consider using k-nearest neighbour to classify the data in Figure 1. If a test point x⋆ was located at
    the centre of the shaded rectangle, describe how the prediction would change for different possible
    k values.                                                                                   (4 marks)
- Question 2(b) (Part B): ([2024 exam PDF](../../references/2024_COMP4702_exam.pdf), Part B)
  - (b) Explain the general relationship between underfitting, overfitting and model complexity in super-
    vised learning. You should refer to Figure 2 to assist with your explanation.          (6 marks)

### 2025 exam

- Question 4. Which of the following is not an example of a hyperparameter? ([2025 exam PDF](../../references/2025_COMP4702_exam.pdf), Part A)
  - (a) The value of k in k-nearest neighbours.
  - (b) The number of layers in a feed-forward neural network.
  - (c) The offset term in linear regression, θ0 .
  - (d) The filter size in a convolutional neural network layer.
- Question 6. A commonly used loss function in machine learning is given by: ([2025 exam PDF](../../references/2025_COMP4702_exam.pdf), Part A)
  - (
      0 if ŷ = y.
    L(y, ŷ) = I{ŷ ̸= y} =
      1 if ŷ ̸= y.
  - This is known as the:
  - (a) Misclassification loss.
  - (b) Mean squared error.
  - (c) ϵ-insensitive loss.
  - (d) Cross-entropy loss.
- Question 1(a)-(c) (Part B): ([2025 exam PDF](../../references/2025_COMP4702_exam.pdf), Part B)
  - (a) Explain the shape of the test curve shown.                                         (4 marks)
  - (b) The training curve in Figure 1 has a misclassification rate of 0 at k = 1. Explain how this is
    possible and hence would be true for any dataset.                                   (4 marks)
  - (c) Figure 1 shows that as k increases, the misclassification rate on the training set increases. What
    does this suggest about the relationship between k, model complexity overfitting and
    underfitting?                                                                             (4 marks)
- Question 2(a)-(c) (Part B): ([2025 exam PDF](../../references/2025_COMP4702_exam.pdf), Part B)
  - (a) Given a test point x′ = [petal length = 3cm, petal width = 2.5cm], what is the predicted output
    from the decision tree and explain how it arrives at this output.                     (5 marks)
  - (b) What is the misclassification rate of the tree?                                      (3 marks)
  - (c) Draw the confusion matrix for this data, including labels for each row and column.    (5 marks)

### How to use these for Week 2 revision

- First pass: answer each question without notes.
- Second pass: map each question to one Chapter 2 skill (`k`-NN intuition, tree split criteria, misclassification/confusion matrix, overfitting-underfitting).
- Third pass: redo any question you missed and write a one-line error diagnosis (concept gap, arithmetic slip, or interpretation mistake).

## Toy example 1: `k`-NN from scratch (classification)

This demonstrates the exam-relevant effect of changing `k`.

```python

def train_test_split(X, y, train_ratio=0.7, seed=4702):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_train = int(train_ratio * len(idx))
    n_train = max(1, min(len(idx) - 1, n_train))
    tr, te = idx[:n_train], idx[n_train:]
    return X[tr], X[te], y[tr], y[te]


def knn_predict_classification(X_train, y_train, X_test, k):
    preds = []
    for x in X_test:
        d = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        nn = np.argsort(d)[:k]
        vote = np.mean(y_train[nn])
        preds.append(1 if vote >= 0.5 else 0)
    return np.array(preds)


rng = np.random.default_rng(SEED)
n = 300
X = rng.normal(0.0, 1.0, size=(n, 2))
# Non-linear boundary: inside circle -> class 1
radius2 = X[:, 0] ** 2 + X[:, 1] ** 2
y = (radius2 < 1.2).astype(int)

# Add mild label noise
flip = rng.random(n) < 0.08
y_noisy = y.copy()
y_noisy[flip] = 1 - y_noisy[flip]

X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, train_ratio=0.7, seed=SEED)

for k in [1, 5, 25]:
    yhat_tr = knn_predict_classification(X_train, y_train, X_train, k)
    yhat_te = knn_predict_classification(X_train, y_train, X_test, k)
    tr_err = np.mean(yhat_tr != y_train)
    te_err = np.mean(yhat_te != y_test)
    print(f"k={k:>2} | train misclass={tr_err:.3f} | test misclass={te_err:.3f}")

print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
```

## Toy example 2: decision-tree split criteria on one candidate node

This is the core exam skill for entropy/Gini/misclassification reasoning.

```python

def impurity_misclassification(p):
    return 1.0 - max(p, 1.0 - p)


def impurity_gini(p):
    return 2.0 * p * (1.0 - p)


def impurity_entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def weighted_split_cost(y_left, y_right, impurity_fn):
    n1, n2 = len(y_left), len(y_right)
    p1 = np.mean(y_left == 1)
    p2 = np.mean(y_right == 1)
    return n1 * impurity_fn(p1) + n2 * impurity_fn(p2)


# Tiny binary dataset with one feature for stump splits
x = np.array([0.1, 0.3, 0.6, 0.9, 1.1, 1.4, 1.7, 2.0])
y = np.array([0,   0,   1,   1,   1,   0,   0,   0])
thresholds = [0.45, 1.0, 1.55]

for t in thresholds:
    left = y[x < t]
    right = y[x >= t]
    c_m = weighted_split_cost(left, right, impurity_misclassification)
    c_g = weighted_split_cost(left, right, impurity_gini)
    c_e = weighted_split_cost(left, right, impurity_entropy)
    print(f"threshold={t:.2f} | misclass_cost={c_m:.3f} | gini_cost={c_g:.3f} | entropy_cost={c_e:.3f}")
```

## Toy example 3: decision stump + confusion matrix

This matches repeated exam-style tasks: prediction path, misclassification rate, confusion matrix.

```python

def fit_decision_stump_1d(x_train, y_train):
    candidates = (x_train[:-1] + x_train[1:]) / 2.0
    best_t, best_cost = None, float("inf")

    for t in candidates:
        left_idx = x_train < t
        right_idx = ~left_idx
        if left_idx.sum() == 0 or right_idx.sum() == 0:
            continue

        left_label = int(np.mean(y_train[left_idx]) >= 0.5)
        right_label = int(np.mean(y_train[right_idx]) >= 0.5)

        yhat = np.where(left_idx, left_label, right_label)
        cost = np.mean(yhat != y_train)
        if cost < best_cost:
            best_cost = cost
            best_t = t
            best_left = left_label
            best_right = right_label

    return best_t, best_left, best_right, best_cost


def stump_predict(x_in, t, left_label, right_label):
    return np.where(x_in < t, left_label, right_label)


def confusion_matrix_binary(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]])  # rows: true [0,1], cols: pred [0,1]


# Reuse a 1D slice for simplicity
rng = np.random.default_rng(SEED)
x_all = np.sort(rng.uniform(-2, 2, size=120))
prob = 1.0 / (1.0 + np.exp(-2.2 * x_all))
y_all = rng.binomial(1, prob)

split = 80
x_tr, y_tr = x_all[:split], y_all[:split]
x_te, y_te = x_all[split:], y_all[split:]

t, left_label, right_label, tr_err = fit_decision_stump_1d(x_tr, y_tr)
yhat_te = stump_predict(x_te, t, left_label, right_label)
te_err = np.mean(yhat_te != y_te)
cm = confusion_matrix_binary(y_te, yhat_te)

print(f"stump threshold={t:.3f}, left_label={left_label}, right_label={right_label}")
print(f"train misclass={tr_err:.3f}, test misclass={te_err:.3f}")
print("confusion matrix [[TN, FP], [FN, TP]]:")
print(cm)
print("Reproducible run check (seed fixed):", SEED)
```

## Week 2 wrap-up (what to remember)

- Chapter 2 teaches supervised learning through two core non-parametric methods: `k`-NN and decision trees.
- The central exam idea is always **generalisation**, not just training fit.
- `k` and tree depth are model-complexity controls that drive overfitting/underfitting behaviour.
- For decision-tree classification splits, be comfortable with misclassification, Gini, and entropy.
- Be ready to compute misclassification rate and confusion matrix by hand from predictions.

## Sources used

- `references/CourseSummaryTable_v1_26.pdf`
- `references/main-text-book-machine-learning-lindholm-2022.pdf` (Chapter 2)
- `references/2023_COMP4702_exam.pdf`
- `references/2024_COMP4702_exam.pdf`
- `references/2025_COMP4702_exam.pdf`
