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

# Week 3 Lecture Notes: Unsupervised, Semi-supervised, and Generative Models

## Contents

- [Scope for Week 3](#scope-for-week-3)
- [Learning goals for this notebook](#learning-goals-for-this-notebook)
- [Chapter 10 summary: what matters most for Week 3](#chapter-10-summary)
  - [10.1 Gaussian mixture models and discriminant analysis](#gmm-and-discriminant-analysis)
  - [Semi-supervised GMM and the EM idea](#semi-supervised-gmm-and-em)
  - [10.2 Cluster analysis](#cluster-analysis)
  - [10.4 Representation learning and dimensionality reduction](#representation-learning-and-dimensionality-reduction)
- [Core comparisons to remember](#core-comparisons)
- [Exam-oriented takeaways](#exam-oriented-takeaways)
- [Past exam questions (2023-2025)](#past-exam-questions)
- [Toy example: PCA on a tiny 2D dataset](#toy-example-pca)
- [Minimal toy example: hard vs soft cluster assignment](#toy-example-hard-vs-soft)
- [Week 3 revision summary](#week-3-revision-summary)

<a id="scope-for-week-3"></a>
## Scope for Week 3

- Topic: unsupervised and semi-supervised learning; introduction to generative models.
- Important concepts: density estimation, clustering, dimensionality reduction.
- Core methods: Gaussian mixture models (GMMs), Expectation-Maximisation (EM), `k`-means, principal component analysis (PCA).
- Reading: [Lindholm (2022), Chapter 10.1, 10.2, 10.4](../../references/main-text-book-machine-learning-lindholm-2022.pdf).
- Schedule note: the course summary also lists `Hand01` Sections 6.4, 9.1, 9.2 as options, but that source is not present locally, so these notes stay grounded in Lindholm and the local course materials.
- Prac alignment: Week 3 prac focuses on unsupervised learning.

<a id="learning-goals-for-this-notebook"></a>
## Learning goals for this notebook

1. Explain what changes when labels are missing and why generative models become useful.
2. Understand the objective, update logic, and failure modes of GMMs, EM, `k`-means, and PCA.
3. Prioritise the Week 3 ideas that have already appeared in 2023-2025 exams.

```python
import random
import numpy as np

SEED = 4702
random.seed(SEED)
np.random.seed(SEED)
print(f"Seed set to {SEED}")
```

<a id="chapter-10-summary"></a>
## Chapter 10 summary: what matters most for Week 3

Week 3 is about learning useful structure from inputs even when labels are partly missing or entirely absent.

- In **supervised learning**, we observe labelled pairs $(x_i, y_i)$ and learn to predict $y$ from $x$.
- In **semi-supervised learning**, only some training points have labels.
- In **unsupervised learning**, we only observe $\{x_i\}_{i=1}^n$ and try to learn structure in the input distribution itself.

The Lindholm chapter ties these together through a generative viewpoint:

- model the joint distribution $p(x, y)$ when labels exist,
- marginalise out the labels to get a model for $p(x)$ when labels are hidden,
- use that model for classification, clustering, or representation learning.

<a id="gmm-and-discriminant-analysis"></a>
## 10.1 Gaussian mixture models and discriminant analysis

### One-sentence definition

A **generative model** learns how the data is distributed, jointly across inputs and outputs $p(\mathbf{x},y)$, while a **discriminative model** learns only how to predict the output from a given input $p(y|\mathbf{x})$.

A **Gaussian mixture model (GMM)** is a generative model that represents the joint distribution as
$$
p(x, y) = p(x \mid y)p(y),
$$
with a Gaussian distribution for each class-conditional input distribution.

### Objective

The GMM assumes
$$
p(y=m) = \pi_m, \qquad
p(x \mid y=m) = \mathcal{N}(x \mid \mu_m, \Sigma_m),
$$
so the full model becomes
$$
p(x, y) = \mathcal{N}(x \mid \mu_y, \Sigma_y)\pi_y.
$$

Why this matters:

- It models the **joint** distribution, not only the decision boundary.
- It gives a direct path from density modelling to classification through Bayes' rule.
- It can use unlabelled data in a way purely discriminative models cannot.

### Supervised learning of the GMM

With fully labelled training data $T=\{(x_i, y_i)\}_{i=1}^n$, Lindholm writes the learning problem as maximising the joint log-likelihood:
$$
\hat{\theta}
= \arg\max_{\theta}\ln p(\{x_i, y_i\}_{i=1}^n \mid \theta).
$$

In the supervised case this has closed-form maximum likelihood estimates:

- class prior: $\hat{\pi}_m = n_m/n$,
- class mean: $\hat{\mu}_m = \frac{1}{n_m}\sum_{i:y_i=m} x_i$,
- class covariance: empirical covariance of class $m$.

This is a useful contrast with later models: no iterative numerical optimisation is required here.

### From GMM to classification: QDA and LDA

Prediction uses Bayes' rule:
$$
p(y \mid x) = \frac{p(x,y)}{p(x)}.
$$

Then classify with the most probable class:
$$
\hat{y}(x_*) = \arg\max_m p(y=m \mid x_*).
$$

Two important special cases:

- **QDA**: each class has its own covariance $\Sigma_m$, so the decision boundary is quadratic in $x$.
- **LDA**: all classes share a common covariance $\Sigma$, so quadratic terms cancel and the decision boundary becomes linear.

### Why generative modelling can break

- The Gaussian assumption may be too crude for real data.
- Covariance estimation can become unstable in high dimensions or with small class sample sizes.
- A model can exploit unlabelled data only if its structural assumptions are at least approximately sensible.

<a id="semi-supervised-gmm-and-em"></a>
## Semi-supervised GMM and the EM idea

### One-sentence definition

Semi-supervised GMM learning uses labelled points exactly where labels are known and soft class probabilities where labels are missing.

### Core setup

Suppose we have

- labelled data $\{(x_i, y_i)\}_{i=1}^{n_l}$,
- unlabelled data $\{x_i\}_{i=n_l+1}^{n}$.

If we ignore the unlabelled points, we waste information about the input distribution. The GMM avoids that by treating missing labels as latent variables.

### EM intuition

The problem no longer has a closed-form MLE, so Lindholm introduces an iterative procedure:

1. Initialise the model using the labelled data.
2. Estimate missing-label probabilities using the current model.
3. Re-estimate parameters using labelled data plus those soft assignments.
4. Repeat until convergence.

The key quantity is the **responsibility** or weight for class $m$ on point $x_i$:

- if $y_i$ is known, use a hard weight consistent with the known class,
- if $y_i$ is missing, use the soft posterior probability $p(y_i=m \mid x_i, \hat{\theta})$.

This is the practical meaning of the EM split:

- **E-step**: infer hidden class memberships probabilistically.
- **M-step**: maximise the expected complete-data log-likelihood given those inferred memberships.

### Why this matters

- It explains how semi-supervised learning can be principled rather than heuristic.
- It foreshadows fully unsupervised clustering by simply removing all labels.
- It is one of the clearest examples of latent-variable learning in the course.

### Failure modes

- EM is a local optimisation method, not a global one.
- Different initialisations can converge to different stationary points.
- The unsupervised GMM likelihood is ill-posed without care: a covariance can collapse around one point and make the likelihood diverge.

Conservative practical takeaway: initialise carefully, run multiple times, and do not over-interpret a single clustering result.

<a id="cluster-analysis"></a>
## 10.2 Cluster analysis

### One-sentence definition

Clustering groups data points so that points in the same cluster are more similar to each other than to points in other clusters.

### Unsupervised GMM

In the unsupervised setting the labels are completely latent, so we model
$$
p(x) = \sum_{m=1}^M \pi_m \, \mathcal{N}(x \mid \mu_m, \Sigma_m).
$$

The objective is maximum likelihood on the observed inputs:
$$
\hat{\theta}
= \arg\max_{\theta}\ln p(\{x_i\}_{i=1}^n \mid \theta).
$$

This is the canonical Week 3 density-estimation view:

- the model learns where data tends to occur,
- cluster assignments are inferred rather than observed,
- uncertainty in assignments is represented probabilistically.

### `k`-means as a simpler clustering method

`k`-means removes the probabilistic machinery and makes hard assignments to clusters.

Its optimisation target can be written as minimising within-cluster squared distances:
$$
\arg\min_{R_1,\dots,R_M}\sum_{m=1}^M \sum_{x \in R_m}\|x-\hat{\mu}_m\|_2^2.
$$

Lloyd's algorithm then alternates:

1. assign each point to the nearest cluster centre,
2. recompute each centre as the mean of its assigned points.

This is one of the highest-yield exam ideas in Week 3:

- GMM + EM uses **soft** assignments,
- `k`-means uses **hard** assignments.

Lindholm also makes two other important distinctions:

- `k`-means uses Euclidean distance directly,
- EM for GMM effectively accounts for cluster covariance structure as well.

### Choosing the number of clusters

The number of clusters $M$ is a model choice, not something the training objective can safely optimise by itself.

Why:

- increasing $M$ always increases model flexibility,
- training loss can keep decreasing even when the solution becomes meaningless,
- the extreme $M=n$ solution is a clustering analogue of overfitting.

For `k`-means, Lindholm highlights the **elbow method**:

- fit models for several values of $M$,
- plot the objective against $M$,
- look for the point where extra clusters stop buying much improvement.

This is only a heuristic. In unsupervised learning, "best loss" and "most useful structure" are not always the same.

<a id="representation-learning-and-dimensionality-reduction"></a>
## 10.4 Representation learning and dimensionality reduction

### One-sentence definition

Dimensionality reduction learns a lower-dimensional representation that preserves as much useful structure as possible.

### Why it matters

High-dimensional data can be:

- expensive to store and process,
- noisy or redundant,
- hard to visualise,
- hard to model reliably with limited data.

Week 3 focuses on **PCA**, which is the linear version of this idea.

### PCA objective and geometry

PCA starts by centring the data:
$$
x_{0,i} = x_i - \bar{x}.
$$

Arrange the centred samples into a data matrix $X_0$. Lindholm then computes an SVD:
$$
X_0 = U\Sigma V^\top.
$$

The principal-component scores are
$$
Z_0 = X_0V = U\Sigma.
$$

Interpretation:

- columns of $V$ are the principal axes,
- the transformed coordinates in $Z_0$ are the principal components or scores,
- keeping only the first $q$ columns gives the best rank-$q$ linear approximation.

This is also why PCA can be viewed as a linear autoencoder with a bottleneck layer.

### Covariance interpretation

The covariance matrix of the centred data is
$$
\frac{1}{n}X_0^\top X_0 = V\Lambda V^\top,
$$
where the eigenvectors give the principal directions and the eigenvalues measure variance captured along those directions.

This produces two common exam statements:

- PCA can be done via SVD of the centred data matrix,
- PCA can also be done via eigendecomposition of the covariance matrix.

### What can break

- PCA is linear, so it cannot represent strongly nonlinear structure.
- Large-scale variables can dominate unless inputs are standardised when appropriate.
- High explained variance does not automatically mean the representation is useful for a downstream supervised task.

<a id="core-comparisons"></a>
## Core comparisons to remember

### GMM vs logistic regression

- GMM is **generative**: models $p(x,y)$.
- Logistic regression is **discriminative**: models $p(y \mid x)$ directly.

### GMM/EM vs `k`-means

- GMM/EM: soft probabilistic assignments.
- `k`-means: hard nearest-centre assignments.
- GMM/EM: covariance-aware clusters.
- `k`-means: Euclidean-distance clusters.

### Clustering vs PCA

- Clustering looks for structure among **rows** (groups of data points).
- PCA looks for structure among **columns/directions** (low-dimensional representation).

<a id="exam-oriented-takeaways"></a>
## Exam-oriented takeaways

Most likely Week 3 concepts to be assessed:

1. explain why `k`-means is a simplified version of EM for a GMM,
2. distinguish hard cluster assignments from soft responsibilities,
3. define a generative model and explain why modelling $p(x)$ helps with missing labels,
4. state what PCA is doing geometrically,
5. explain what PCA eigenvalues measure,
6. identify the role of initialisation and local optima in EM and `k`-means,
7. explain why choosing the number of clusters is a model-selection problem.

Fast exam checklist:

- Can I write down $p(x,y)=p(x\mid y)p(y)$ and explain why that is generative?
- Can I explain the E-step and M-step without getting lost in symbols?
- Can I state the exact hard-vs-soft assignment difference between `k`-means and GMMs?
- Can I explain PCA as "rotate/project onto directions of largest variance"?
- Can I say what the PCA eigenvalues correspond to?

<a id="past-exam-questions"></a>
## Past exam questions (2023-2025)

These are copied from [COMP4702_exams_2023_2025.md](../../references/COMP4702_exams_2023_2025.md), selected using [exam_questions_2023_2025_by_week.csv](../../references/exam_questions_2023_2025_by_week.csv).

### 2023 Part A, Question 9

Source: [2023 exam PDF](../../references/2023_COMP4702_exam.pdf)

- What is the main difference between the `k`-means clustering algorithm and Gaussian mixture models (trained using the EM algorithm)?
- (a) `k`-means clustering can handle data with missing values, while Gaussian mixture models cannot.
- (b) `k`-means clustering uses a "hard" assignment of data points to clusters, whereas Gaussian mixture models use a "soft" probabilistic assignment.
- (c) The EM algorithm is prone to getting stuck in local minima, whereas the `k`-means algorithm always converges to the global optimum.
- (d) `k`-means clustering requires a technique for initialization whereas the EM algorithm does this automatically.

### 2023 Part A, Question 10

Source: [2023 exam PDF](../../references/2023_COMP4702_exam.pdf)

- Which of the following statements about Principal Component Analysis (PCA) is incorrect?
- (a) Given a dataset with `p` features, PCA can reduce the dimensionality of the data to a minimum of `p/2` features.
- (b) PCA performs dimensionality reduction via a linear projection of the data.
- (c) PCA can be carried out by calculating a singular value decomposition of the centered data matrix.
- (d) PCA can be carried out by calculating the eigenvectors and eigenvalues of the covariance matrix of the data.

### 2024 Part B, Question 6(b)

Source: [2024 exam PDF](../../references/2024_COMP4702_exam.pdf)

- `k`-means is a well-known clustering algorithm. Explain briefly how it works (i.e. the two main steps involved) and how this can be seen as a simplified version of the expectation-maximization (EM) algorithm for fitting a Gaussian mixture model. (6 marks)

### 2025 Part A, Question 2

Source: [2025 exam PDF](../../references/2025_COMP4702_exam.pdf)

- The `k`-Means clustering algorithm can be seen as a simplified case of which other algorithm that we have studied during the course:
- (a) Stochastic gradient descent.
- (b) Principal component analysis.
- (c) Recursive binary splitting.
- (d) The Expectation Maximisation algorithm.

### 2025 Part A, Question 9

Source: [2025 exam PDF](../../references/2025_COMP4702_exam.pdf)

- Principal component analysis (PCA) can be performed by calculating the eigenvectors and eigenvalues of the covariance matrix of a data set. The eigenvalues measure:
- (a) The mean squared error of the model on the data.
- (b) The fraction of variance captured by the respective principal components.
- (c) The amount of memory required to store the data after performing dimensionality reduction.
- (d) The angle of rotation of each principal component compared to the original coordinate system.

### 2025 Part B, Question 3

Source: [2025 exam PDF](../../references/2025_COMP4702_exam.pdf)

- Given a labeled dataset, a Gaussian mixture model can be trained to perform classification by fitting the parameters of the model using maximum likelihood estimates. This is a form of supervised learning, since the class labels are used. Assume that an additional set of unlabelled data becomes available, so that we now have:
$$
T = \{\{(x_i, y_i)\}_{i=1}^{n_l}, \{x_i\}_{i=n_l+1}^{n}\}
$$
- Explain how learning in the Gaussian mixture model can be modified to perform semi-supervised learning using all of the data available, $T$. You should use the notation given to assist with your explanation. You may also use pseudocode or a diagram but this is not necessary. (6 marks)

### Short answer key for revision

- 2023 Part A Q9: **(b)**
- 2023 Part A Q10: **(a)**
- 2025 Part A Q2: **(d)**
- 2025 Part A Q9: **(b)**

<a id="toy-example-pca"></a>
## Toy example: PCA on a tiny 2D dataset

This example keeps the algebra small enough to inspect directly.

```python
X = np.array(
    [
        [2.0, 1.0],
        [3.0, 2.0],
        [4.0, 2.5],
        [5.0, 4.0],
        [6.0, 4.5],
    ],
    dtype=float,
)

x_bar = X.mean(axis=0, keepdims=True)
X0 = X - x_bar

U, s, VT = np.linalg.svd(X0, full_matrices=False)
V = VT.T
Z0 = X0 @ V
explained_var = (s**2) / X.shape[0]
explained_ratio = explained_var / explained_var.sum()

print("Mean:")
print(x_bar)
print("\nPrincipal axes V:")
print(np.round(V, 4))
print("\nScores Z0:")
print(np.round(Z0, 4))
print("\nExplained variance ratio:")
print(np.round(explained_ratio, 4))
```

### What to check

- The first principal axis should align with the main elongated direction of the data cloud.
- The first explained-variance ratio should be close to 1 because the points lie close to a line.
- If you keep only the first principal component, you get a strong 1D summary with little information loss.

<a id="toy-example-hard-vs-soft"></a>
## Minimal toy example: hard vs soft cluster assignment

Suppose a point lies between two cluster centres:

- `k`-means must assign it fully to one cluster,
- a GMM can assign, for example, 0.55 responsibility to cluster 1 and 0.45 to cluster 2.

That single difference is the core conceptual bridge from `k`-means to EM.

<a id="week-3-revision-summary"></a>
## Week 3 revision summary

- GMMs model distributions; they do not only draw boundaries.
- EM handles missing latent labels by alternating inference and parameter re-estimation.
- `k`-means is the hard-assignment simplification you should immediately compare against EM.
- PCA is linear dimensionality reduction via variance-maximising directions and projection.
- Week 3 exam answers are often short but hinge on precise wording: hard vs soft, generative vs discriminative, variance vs error, local optimum vs global optimum.
