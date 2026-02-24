# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,md,py:percent
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Week 1 Lecture Notes: Intro + Lindholm Chapter 1
#
# ## Scope for Week 1
#
# - Topic: Introduction and background review.
# - Important concept: machine learning definitions and examples.
# - Reading: [Lindholm (2022), Chapter 1](../../references/main-text-book-machine-learning-lindholm-2022.pdf).
# - Prac: no prac this week; focus on background knowledge.
#
# ## Learning goals for this notebook
#
# 1. Understand the key machine learning ideas introduced in Chapter 1.
# 2. Focus on concepts likely to be assessed in COMP4702 exams.
# 3. Build intuition with toy Python examples that are small and reproducible.

# %%
import random
import numpy as np

SEED = 4702
random.seed(SEED)
np.random.seed(SEED)
print(f"Seed set to {SEED}")

# %% [markdown]
# ## Week 1 lecture transcript summary (admin + course framing)
#
# Based on lecture transcript: [COMP4702_S1_2026_ST-transcript.txt](COMP4702_S1_2026_ST-transcript.txt)
#
# ### Key logistics
#
# - Week 1 is mostly course admin and course expectations; only a light start on ML content.
# - No prac classes in Week 1.
# - Use `Ed` as the default channel for questions (large class size; email does not scale well).
# - The course is designed to be interactive where possible, even in a large class.
#
# ### Assessment structure (high-level)
#
# - Final exam is the major component (pen-and-paper, closed-book).
# - There is one assignment, and the rubric is being adapted because LLM tools can automate much of the old workflow.
# - There is a choice for the smaller assessment component:
#   - lecture participation, or
#   - prac demos.
# - `COMP4702` and `COMP7703` share core content but may differ in assignment rubric/assessment details.
#
# ### AI / LLM policy takeaways
#
# - The lecturer does not treat AI bans as a practical solution in all settings.
# - AI use may be allowed in some coursework contexts, but not relied on to assess learning.
# - Final exam and oral/prac demo components help assess individual understanding.
# - Main message: use LLMs carefully so they do not replace the learning process.
#
# ### Learning philosophy (useful for doing well in the course)
#
# - Focus on learning, not only grades.
# - Engage actively in class discussion and on `Ed`.
# - Write notes by hand (or on a tablet) during class to support learning.
# - Expect uncertainty, trade-offs, and “it depends” answers in ML rather than one universally best method.
#
# ### Chapter 1 tie-in from the lecture
#
# The lecturer previews Lindholm Chapter 1 by emphasising:
#
# - ML as learning from data using a mathematical model and learning algorithm,
# - parameter adjustment as the core learning process (often via optimisation),
# - classification vs regression as core supervised-learning problem types,
# - uncertainty/probabilistic outputs as a recurring theme,
# - real-world examples (ECG diagnosis, crystal energy prediction, soccer goal probability, image pixel labels, pollution estimation) as motivation.
#
# ### Week 1 exam-relevant takeaways from the lecture
#
# - Know the distinction between:
#   - mathematical model (what is being fitted), and
#   - learning algorithm (how it is fitted).
# - Be comfortable with:
#   - classification vs regression,
#   - labels/training data,
#   - generalisation to unseen data,
#   - uncertainty in model predictions.
#
# ## Chapter 1 summary (Lindholm): what matters for ML
#
# ### 1) What machine learning is (core framing)
#
# Machine learning is about learning, reasoning, and acting based on data. The difference from data analysis is that machine learning uses an automated process and a computer program is learned from the data. Another way to say this is that machine learning is programming by example.
#
# Machine learning is framed around three cornerstones:
#
# 1. data,
# 2. mathematical model,
# 3. learning algorithm.
#
# The model parameters are adjusted from data using a learning algorithm so predictions generalise to unseen points.
#
# ### 2) Supervised learning and task types
#
# Chapter 1 introduces supervised learning through real examples.
#
# Supervised learning is training a model from labelled data points, where the "supervisor" is either a domain expert or a natural outcome.
#
# These ML tasks can be divided into two distinct problems:
#
# - classification: output is a discrete label,
# - regression: output is a continuous numeric value.
#
# The data is typically split conceptually into:
#
# - labelled training data (used to fit model parameters),
# - unseen data in deployment (used to test whether learning generalises).
#
# Training data consists of:
#
# - input: features / observations provided to the model,
# - output: the label (classification problems) or target value (regression problems).
#
# Labels are given by:
#
# - domain experts (e.g. a doctor assigning labels to an ECG), or
# - natural outcomes (e.g. a soccer match result, or the formation energy computed by DFT for crystals).
#
# ![Lindholm Chapter 1 supervised learning process](lindholm-ch01-supervised-learning.png)
#
# This figure shows the supervised learning workflow: labelled training data is used by a learning algorithm to fit/update a model, then the trained model is used to make predictions on unseen data.
#
# ### 3) Why probability appears early
#
# A key chapter message is that probabilistic predictions are often required.
# Even when there is a true label/value, uncertainty arises from finite data, model assumptions, and noise.
#
# ### 4) Example classes from the chapter (high-level)
#
# - ECG abnormality diagnosis: supervised classification with the model using a deep neural network.
# - Crystal formation energy prediction: supervised regression.
# - Soccer goal probability from shot context: probabilistic classification using a logistic regression model (see Chapter 3).
# - Pixel-wise class prediction (semantic segmentation): classification problem using a deep neural network model.
# - Spatio-temporal pollution estimation/forecasting: richer prediction settings beyond scalar outputs.
#
# ### 5) What can fail
#
# - weak generalisation (good training fit, poor unseen performance),
# - inadequate features/inputs for the true process,
# - mismatch between model assumptions and data,
# - over-confidence when uncertainty is ignored.
#
# ## Exam-oriented takeaways (based on 2023, 2024, 2025 exams)
#
# From prior exams, Week 1-style ideas repeatedly show up as MCQ fundamentals and as conceptual setup in longer questions.
#
# Most likely assessable concepts:
#
# 1. define supervised learning clearly,
# 2. distinguish classification vs regression,
# 3. explain training/validation/test purpose,
# 4. describe generalisation and overfitting at a conceptual level,
# 5. identify what a model output means at a decision boundary (e.g., logistic output 0.5),
# 6. identify hyperparameters vs learned parameters.
#
# Fast exam checklist:
#
# - Can I state the supervised learning objective in one sentence?
# - Can I classify a task as classification/regression quickly?
# - Can I explain why low training error alone is insufficient?
# - Can I explain when probability outputs are more appropriate than hard labels?
#
# ## Past Exam Questions (2023-2025, paraphrased)
#
# Note: These are paraphrased study prompts based on the last three exam papers. Use the linked PDFs for the exact wording.
#
# ### 2023 exam (Week 1-relevant prompts)
#
# - MCQ 1 (Part A): Which option correctly defines supervised learning? ([2023 exam PDF](../../references/2023_COMP4702_exam.pdf), Part A)
#   - (a) A type of machine learning where the algorithm learns to optimize a reward
#     function through trial and error.
#   - (b) A type of machine learning where the algorithm learns to identify patterns
#     and relationships in unlabeled data.
#   - (c) A type of machine learning where the algorithm learns to make predictions
#     based on input data and output labels.
#   - (d) A type of machine learning where the algorithm learns to cluster data points
#     based on their similarity.
# - MCQ 3 (Part A): What is the purpose of a hold-out dataset? ([2023 exam PDF](../../references/2023_COMP4702_exam.pdf), Part A)
#   - (a) To train the model on new data that it hasn’t seen before.
#   - (b) To estimate the accuracy of the model on data that it has not been trained on.
#   - (c) To fine-tune the model’s hyperparameters and avoid overfitting.
#   - (d) To estimate the accuracy of the model on data that it has already been
#     trained on.
#
# ### 2024 exam (Week 1-relevant prompts)
#
# - MCQ 1 (Part A): Which statement about machine learning is incorrect? (tests general
#   ML definitions and goals such as generalisation and optimisation). ([2024 exam PDF](../../references/2024_COMP4702_exam.pdf), Part A)
#   - (a) Machine learning uses a set of data to determine suitable parameters for a
#     mathematical model.
#   - (b) Supervised learning means that the machine learning model uses the previous
#     error of the model to predict the future error.
#   - (c) The goal of machine learning is to achieve good generalisation performance.
#   - (d) In machine learning, the training algorithm typically solves an optimisation
#     problem.
#
# ### 2025 exam (Week 1-relevant prompts)
#
# - No clearly relevant Chapter 1 / Week 1-focused questions identified in Part A. ([2025 exam PDF](../../references/2025_COMP4702_exam.pdf))
#
# ### How to use these for Week 1 revision
#
# - First pass: answer from memory in one sentence each.
# - Second pass: classify each as definition / data representation / evaluation /
#   optimisation knowledge.
# - Third pass: connect each question back to Lindholm Chapter 1 terminology (data, model, learning algorithm, generalisation, uncertainty).
#
# ### Past exam answers
# - 2023 Part A Q1: c, Q3: b
# - 2024 Part A Q1: b
#
# ## Toy example 1: classification (ECG-style binary risk flag)
#
# This is a tiny synthetic analogue of the Chapter 1 ECG story.

# %%
import numpy as np


def train_test_split(X, y, train_ratio=0.8, seed=4702):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_train = int(train_ratio * len(idx))
    tr, te = idx[:n_train], idx[n_train:]
    return X[tr], X[te], y[tr], y[te]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def log_loss(y_true, y_prob, eps=1e-9):
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))


def fit_logistic_gd(X, y, lr=0.1, n_steps=1000):
    n, p = X.shape
    w = np.zeros(p)
    b = 0.0
    losses = []
    for _ in range(n_steps):
        logits = X @ w + b
        probs = sigmoid(logits)
        losses.append(log_loss(y, probs))
        grad_w = (X.T @ (probs - y)) / n
        grad_b = np.mean(probs - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, np.array(losses)


rng = np.random.default_rng(SEED)
n = 400
hr_irregularity = rng.normal(loc=0.0, scale=1.0, size=n)
qrs_width = rng.normal(loc=0.0, scale=1.0, size=n)
X = np.column_stack([hr_irregularity, qrs_width])

# Synthetic latent rule: higher irregularity and wider QRS => higher abnormality risk
logit_true = 1.4 * hr_irregularity + 1.0 * qrs_width - 0.2
p_true = sigmoid(logit_true)
y = rng.binomial(n=1, p=p_true).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.8, seed=SEED)
w, b, losses = fit_logistic_gd(X_train, y_train, lr=0.2, n_steps=1200)

train_prob = sigmoid(X_train @ w + b)
test_prob = sigmoid(X_test @ w + b)
train_pred = (train_prob >= 0.5).astype(float)
test_pred = (test_prob >= 0.5).astype(float)

train_acc = np.mean(train_pred == y_train)
test_acc = np.mean(test_pred == y_test)

print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
print("Initial loss:", round(float(losses[0]), 4), "Final loss:", round(float(losses[-1]), 4))
print("Loss decreased:", bool(losses[-1] < losses[0]))
print("Train accuracy:", round(float(train_acc), 3), "Test accuracy:", round(float(test_acc), 3))

# %% [markdown]
# ## Toy example 2: regression (formation-energy style)
#
# This is a simple supervised regression setup with a known target function.

# %%
rng = np.random.default_rng(SEED)
n = 300
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
X = np.column_stack([x1, x2])
noise = rng.normal(0, 0.3, n)

# Synthetic "formation energy" target
y = 2.0 * x1 - 1.2 * x2 + 0.5 + noise

X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.8, seed=SEED)

# Closed-form linear regression with bias term
X_train_aug = np.column_stack([np.ones(X_train.shape[0]), X_train])
X_test_aug = np.column_stack([np.ones(X_test.shape[0]), X_test])
theta = np.linalg.pinv(X_train_aug.T @ X_train_aug) @ X_train_aug.T @ y_train

train_hat = X_train_aug @ theta
test_hat = X_test_aug @ theta

train_rmse = float(np.sqrt(np.mean((train_hat - y_train) ** 2)))
test_rmse = float(np.sqrt(np.mean((test_hat - y_test) ** 2)))

print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
print("Estimated parameters [bias, w1, w2]:", np.round(theta, 3))
print("Train RMSE:", round(train_rmse, 3), "Test RMSE:", round(test_rmse, 3))

# %% [markdown]
# ## Toy example 3: probabilistic output (soccer-shot style)
#
# The learning goal is: sometimes we want probability of success, not only yes/no prediction.

# %%
rng = np.random.default_rng(SEED)
n = 500

# distance to goal in meters, shot angle in radians
distance = rng.uniform(2.0, 35.0, n)
angle = rng.uniform(0.05, 1.2, n)

# Larger angle helps, larger distance hurts
logit = 2.2 * angle - 0.18 * distance + 1.1
p_goal = sigmoid(logit)
y_goal = rng.binomial(1, p_goal, size=n).astype(float)

X = np.column_stack([distance, angle])
X_train, X_test, y_train, y_test = train_test_split(X, y_goal, train_ratio=0.8, seed=SEED)

w, b, losses = fit_logistic_gd(X_train, y_train, lr=0.05, n_steps=1500)
probs_test = sigmoid(X_test @ w + b)
preds_test = (probs_test >= 0.5).astype(float)

accuracy = float(np.mean(preds_test == y_test))
brier = float(np.mean((probs_test - y_test) ** 2))

print("Final train loss:", round(float(losses[-1]), 4))
print("Loss decreased:", bool(losses[-1] < losses[0]))
print("Test accuracy:", round(accuracy, 3))
print("Test Brier score (lower is better):", round(brier, 3))
print("Decision boundary probability threshold for logistic model:", 0.5)

# %% [markdown]
# ## Week 1 wrap-up (what to remember)
#
# - Machine learning = data + model + learning algorithm.
# - For supervised learning, always state input, output, and prediction goal.
# - Classification and regression are the core Week 1 problem types.
# - Generalisation (not just training fit) is the real objective.
# - Probabilistic predictions are often more useful and realistic.
#
# ## Sources used
#
# - [Course summary table PDF](../../references/CourseSummaryTable_v1_26.pdf)
# - [Lindholm textbook PDF](../../references/main-text-book-machine-learning-lindholm-2022.pdf) (Chapter 1)
# - [2023 exam PDF](../../references/2023_COMP4702_exam.pdf)
# - [2024 exam PDF](../../references/2024_COMP4702_exam.pdf)
# - [2025 exam PDF](../../references/2025_COMP4702_exam.pdf)
