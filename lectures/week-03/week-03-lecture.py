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
# # Notebook
#
# Created by scripts/setup_notebook_workflow.sh

# %%
import random
import numpy as np

SEED = 4702
random.seed(SEED)
np.random.seed(SEED)
print(f"Seed set to {SEED}")

# %% [markdown]
# ## Lindholm (2022) Chapter 10.1 Summary
#
# ### Section title
#
# **10.1 The Gaussian Mixture Model (GMM) and Discriminant Analysis**
#
# ### One-line summary
#
# Section 10.1 introduces a **generative classifier** that models the joint distribution
# $p(\mathbf{x}, y)$ using class-conditional Gaussians, and shows how this yields **QDA**,
# **LDA**, and a principled path to **semi-supervised learning**.
#
# ### Core idea: generative vs discriminative
#
# - Earlier classifiers (e.g. logistic regression) model $p(y \mid \mathbf{x})$ directly.
# - The GMM models the joint distribution via
#   $p(\mathbf{x}, y) = p(\mathbf{x}\mid y)p(y)$.
# - This is more restrictive (more assumptions) but also richer, because it models the
#   input distribution and can use unlabelled data.
#
# ### GMM model assumptions (classification setting)
#
# Assume:
# - $\mathbf{x}$ is numerical
# - $y \in \{1, \dots, M\}$ is categorical
#
# Model:
# - $p(y=m) = \pi_m$ (categorical class prior)
# - $p(\mathbf{x}\mid y=m) = \mathcal{N}(\mathbf{x}\mid \boldsymbol{\mu}_m, \mathbf{\Sigma}_m)$
#
# Interpretation:
# - Each class has its own Gaussian in feature space.
# - Marginally, $p(\mathbf{x})$ becomes a mixture of Gaussians (hence “GMM”).
#
# ### Supervised learning of the GMM (closed-form MLE)
#
# Given labelled data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$, the parameters are estimated by
# maximum likelihood on the **joint** likelihood.
#
# Closed-form estimates:
# - Class prior: $\hat{\pi}_m = n_m/n$
# - Class mean: $\hat{\boldsymbol{\mu}}_m =$ empirical mean of class $m$
# - Class covariance: $\hat{\mathbf{\Sigma}}_m =$ empirical covariance of class $m$
#
# Key point:
# - No numerical optimisation is needed in the supervised case.
# - Even if the Gaussian assumption is imperfect, the resulting classifiers can still work well.
#
# ### From GMM to classification: QDA
#
# Use Bayes' rule to compute
# $p(y=m\mid \mathbf{x}_*) \propto \hat{\pi}_m \,\mathcal{N}(\mathbf{x}_* \mid \hat{\boldsymbol{\mu}}_m, \hat{\mathbf{\Sigma}}_m)$
# and predict the most probable class.
#
# Because $\log \mathcal{N}(\cdot)$ is quadratic in $\mathbf{x}$:
# - decision boundaries are **quadratic**
# - this gives **Quadratic Discriminant Analysis (QDA)**
#
# ### LDA as a special case
#
# If all classes share the same covariance:
# - $\mathbf{\Sigma}_1 = \cdots = \mathbf{\Sigma}_M = \mathbf{\Sigma}$
#
# then quadratic terms cancel in the discriminant score, so:
# - decision boundaries become **linear**
# - this gives **Linear Discriminant Analysis (LDA)**
#
# Practical note:
# - LDA and logistic regression are both linear classifiers and often perform similarly,
#   but they are learned differently (generative vs discriminative).
#
# ### Semi-supervised GMM (partially labelled data)
#
# Setting:
# - Some training points have labels, many do not.
# - Throwing away unlabelled data is simple but can waste information.
#
# Why GMM helps:
# - Since the model includes $p(\mathbf{x})$, unlabelled inputs can inform the class
#   structure (clusters) and improve estimates of $p(\mathbf{x}\mid y)$.
#
# ### Semi-supervised learning procedure (EM-style)
#
# The semi-supervised maximum-likelihood problem has no closed-form solution, so the
# section proposes an iterative procedure:
#
# 1. Fit an initial GMM using only labelled data.
# 2. Predict class probabilities for unlabelled points using current parameters.
# 3. Re-estimate GMM parameters using labelled data + soft class weights for unlabelled data.
# 4. Repeat until convergence.
#
# Important detail:
# - Use **soft probabilities** $p(y=m\mid \mathbf{x}_i, \hat{\theta})$ (not hard labels) for
#   missing labels.
# - Parameter updates become **weighted** versions of the supervised estimates.
#
# This is later identified as an instance of the **Expectation-Maximisation (EM)** algorithm.
#
# ### Why this section matters (exam/understanding)
#
# - Connects **probability modelling** to **classification** (Bayes rule -> QDA/LDA).
# - Shows how modelling assumptions determine boundary shape (quadratic vs linear).
# - Motivates why **generative models** are useful for **semi-supervised learning**.
# - Sets up Section 10.2 (clustering / fully unlabelled case) and EM in more detail.
#
# ### Failure modes / assumptions to watch
#
# - Gaussian class-conditional assumption may be wrong.
# - In high-dimensional data (especially images), a simple Gaussian may be too crude.
# - Generative models can exploit unlabelled data, but bad assumptions can mislead learning.
#
# ### Quick mental checklist
#
# - Objective: model $p(\mathbf{x}, y)$ and use Bayes rule for prediction
# - Optimisation (supervised): closed-form MLE
# - Optimisation (semi-supervised): iterative EM-style weighted updates
# - Boundary type: QDA (class-specific covariance), LDA (shared covariance)
