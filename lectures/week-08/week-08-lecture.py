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
# # Week 8 Lecture Notes: Neural Networks and Deep Learning (CNNs + Dropout)
#
# ## Contents
#
# - [Scope for Week 8](#scope-for-week-8)
# - [Learning goals for this notebook](#learning-goals-for-this-notebook)
# - [Chapter 6 summary](#chapter-6-summary)
#   - [Why dense networks are not ideal for images](#why-dense-networks-are-not-ideal-for-images)
#   - [Convolutional layers: sparse interactions and parameter sharing](#convolutional-layers-sparse-interactions-and-parameter-sharing)
#   - [Stride and zero-padding](#stride-and-zero-padding)
#   - [Pooling layers](#pooling-layers)
#   - [Multiple channels and deeper CNNs](#multiple-channels-and-deeper-cnns)
#   - [Dropout as regularisation](#dropout-as-regularisation)
#   - [Objective, optimisation, failure modes](#objective-optimisation-failure-modes)
# - [Exam-oriented takeaways](#exam-oriented-takeaways)
# - [Past exam questions (2023-2025)](#past-exam-questions-2023-2025)
# - [Toy example 1: convolution, stride, and max pooling by hand](#toy-example-1-convolution-stride-and-max-pooling-by-hand)
# - [Toy example 2: CNN output shapes and parameter counts](#toy-example-2-cnn-output-shapes-and-parameter-counts)
# - [Toy example 3: dropout masks and test-time scaling](#toy-example-3-dropout-masks-and-test-time-scaling)
# - [Week 8 wrap-up](#week-8-wrap-up)
# - [Sources used](#sources-used)
#
# ## Scope for Week 8
#
# - Topic: neural networks and deep learning, with Week 8 focused on convolutional neural networks and dropout.
# - Important concepts: image representation on grids, convolutional filters, sparse interactions, parameter sharing, stride, zero-padding, pooling, multiple channels, and dropout.
# - Algorithms and methods: convolutional neural networks (CNNs), max/average pooling, stochastic-gradient training of CNNs, and dropout regularisation.
# - Reading: [Lindholm (2022), Chapter 6](../../references/main-text-book-machine-learning-lindholm-2022.pdf), with this notebook focusing on Sections 6.3-6.4 because the [Course Summary Table](../../references/CourseSummaryTable_v1_26.pdf) schedules Week 8 as "Neural Networks and Deep Learning" with [Prac W8 – Convolutional networks](../../references/CourseSummaryTable_v1_26.pdf).
# - Scope split note: [Week 7](../week-07/week-07-lecture.md) already covered the dense feedforward / backpropagation part of Chapter 6, so this notebook concentrates on the image-specific and regularisation parts that naturally follow.
# - Supporting-source note: the local [MATLAB notes](../../references/lecture_notes_matlab_2026_v2.pdf) are available, but the structure here is grounded primarily in the course schedule, Lindholm Chapter 6, and the indexed exam materials.
#
# ## Learning goals for this notebook
#
# 1. Explain why CNNs are built differently from fully connected networks when the input is image-like grid data.
# 2. Write down the core CNN operations mathematically and compute output sizes for convolution, stride, padding, and pooling.
# 3. Explain dropout in exam language: what it is trying to achieve, how it is applied during training, and why the test-time weights are rescaled.

# %%
import random
import numpy as np
import matplotlib.pyplot as plt

SEED = 4702
random.seed(SEED)
np.random.seed(SEED)
plt.rcParams["figure.dpi"] = 120

print(f"Seed set to {SEED}")

# %% [markdown]
# ## Chapter 6 summary
#
# Week 8 is the CNN and regularisation half of the neural-network material. The core frame remains the same as elsewhere in the course:
#
# - **Objective:** learn a useful input-output mapping that generalises, now with an architecture that respects spatial structure in images.
# - **Optimisation:** train the parameters with mini-batch gradient descent and backpropagation, exactly as in Week 7.
# - **Failure modes:** too many parameters, weak translation handling, shape mistakes, poor hyperparameter choices, and overfitting.
#
# ### Why dense networks are not ideal for images
#
# Lindholm starts the CNN section by representing a greyscale image as a matrix of pixel intensities, with input variables indexed by position:
#
# - an image is not just a bag of numbers;
# - nearby pixels tend to be more related than distant pixels;
# - flattening the image into one long vector destroys some of that local structure.
#
# This is the motivation for a CNN. A standard dense layer gives every hidden unit access to every pixel, which is often too flexible for image data. In exam terms, the issue is not that dense layers are impossible, but that they ignore useful spatial inductive bias and usually require far more parameters.
#
# Minimal definition:
#
# - a **convolutional neural network** is a neural network designed for grid-structured data, where hidden units depend only on local regions and reuse the same filter weights across positions.
#
# Why it matters:
#
# - the model becomes much more parameter-efficient;
# - local visual patterns such as edges or corners can be detected wherever they appear;
# - the network becomes relatively insensitive to small translations.
#
# ### Convolutional layers: sparse interactions and parameter sharing
#
# The two textbook ideas that define a convolutional layer are:
#
# 1. **Sparse interactions:** each hidden unit only looks at a small local region of the input.
# 2. **Parameter sharing:** the same set of weights is reused across all positions.
#
# For a filter of size $F \times F$, Lindholm writes the convolutional layer as
#
# $$
# q_{ij} = h\left(\sum_{k=1}^{F}\sum_{\ell=1}^{F} x_{i+k-1, j+\ell-1} W_{k,\ell}\right). \qquad (6.29)
# $$
#
# Interpretation:
#
# - $x_{i,j}$ is the zero-padded input image;
# - $W_{k,\ell}$ is the filter;
# - $q_{ij}$ is the output at spatial position $(i,j)$;
# - $h(\cdot)$ is the activation function, often ReLU.
#
# This is the most exam-relevant conceptual contrast with a dense layer:
#
# - dense layer: one unique parameter per connection;
# - convolutional layer: one small filter reused everywhere.
#
# Lindholm's parameter-count example is worth remembering. A $3 \times 3$ filter requires only $3 \cdot 3 + 1 = 10$ parameters including bias, whereas a comparable dense mapping on a $6 \times 6$ image would need far more. This is one of the standard reasons CNNs generalise better on images than a similarly expressive dense network.
#
# Minimal toy example:
#
# - if a filter becomes sensitive to a vertical edge, it can detect that edge near the top-left or near the bottom-right using the same weights.
#
# ### Stride and zero-padding
#
# Two details repeatedly appear in exams: **padding** and **stride**.
#
# **Zero-padding** means adding zeros around the border of the image so that border pixels can still participate in local receptive fields and, in some settings, so that output size is preserved.
#
# **Stride** means the filter does not shift by one pixel each time. If the stride is $s$, the filter moves by $s$ pixels horizontally and vertically.
#
# Lindholm writes the strided convolution as
#
# $$
# q_{ij} = h\left(\sum_{k=1}^{F}\sum_{\ell=1}^{F} x_{s(i-1)+k,\, s(j-1)+\ell} W_{k,\ell}\right). \qquad (6.30)
# $$
#
# Key consequence:
#
# - increasing stride reduces the spatial size of the output without increasing the number of parameters in the filter.
#
# Exam habit to build:
#
# - separate **output size** from **parameter count**.
# - stride changes output dimensions and computation cost.
# - stride does **not** create extra trainable parameters.
#
# For the common 2D size calculation,
#
# $$
# \text{output size per dimension} =
# \left\lfloor \frac{N + 2P - F}{S} \right\rfloor + 1,
# $$
#
# where $N$ is the input size, $P$ is zero-padding, $F$ is filter size, and $S$ is stride.
#
# Failure mode:
#
# - many exam mistakes come from forgetting that padding changes effective input size before the filter is applied.
#
# ### Pooling layers
#
# A pooling layer is another way to reduce spatial resolution after convolution. It summarises a local region but has **no trainable parameters**.
#
# Lindholm gives average pooling as
#
# $$
# \tilde{q}_{ij} = \frac{1}{F^2}\sum_{k=1}^{F}\sum_{\ell=1}^{F} q_{s(i-1)+k,\, s(j-1)+\ell}. \qquad (6.31)
# $$
#
# For **max pooling**, the average is replaced by the maximum over the same local region.
#
# Why pooling matters:
#
# - it condenses information;
# - it can make the representation more invariant to small translations;
# - it reduces later computation.
#
# Textbook contrast worth remembering:
#
# - a strided convolution is cheaper than doing a stride-1 convolution and then pooling;
# - pooling can be more translation-invariant because small shifts often do not change the pooled maximum.
#
# Minimal exam statement:
#
# - pooling changes the representation size but does not introduce extra learned weights.
#
# ### Multiple channels and deeper CNNs
#
# One filter is rarely enough. CNNs therefore use multiple filters, producing multiple **channels**.
#
# Lindholm's structure is:
#
# - one filter produces one output channel;
# - stacking filters produces a tensor of hidden units with dimensions `(rows × columns × channels)`;
# - deeper convolutional layers operate over all channels from the previous layer, not just one image plane.
#
# This matters for parameter counting:
#
# - first convolutional layer on a greyscale image with $C_{\text{in}} = 1$:
#   - parameters per filter = $F \times F \times 1 + 1$;
# - later convolutional layer with $C_{\text{in}}$ input channels and $C_{\text{out}}$ filters:
#   - total parameters = $(F \times F \times C_{\text{in}} + 1) \times C_{\text{out}}$.
#
# As depth increases:
#
# - early layers often learn local low-level features such as edges;
# - later layers combine those into more abstract features;
# - dense layers near the end map the extracted features to logits, then softmax turns those logits into class probabilities.
#
# ### Dropout as regularisation
#
# Section 6.4 frames dropout as a practical, bagging-like way to reduce variance and overfitting in neural networks without training a completely separate ensemble of full models.
#
# Core idea:
#
# - randomly drop some units during training;
# - this creates many random sub-networks that share parameters;
# - shared-parameter ensembling reduces the computational burden relative to ordinary bagging.
#
# For layer $\ell-1$, Lindholm writes dropout as
#
# $$
# m_j^{(\ell-1)} =
# \begin{cases}
# 1 & \text{with probability } r, \\
# 0 & \text{with probability } 1-r,
# \end{cases}
# \qquad (6.33a)
# $$
#
# $$
# \tilde{q}^{(\ell-1)} = m^{(\ell-1)} \odot q^{(\ell-1)}, \qquad (6.33b)
# $$
#
# $$
# q^{(\ell)} = h\left(W^{(\ell)} \tilde{q}^{(\ell-1)} + b^{(\ell)}\right). \qquad (6.33c)
# $$
#
# Here $r$ is the keep probability.
#
# Interpretation:
#
# - each gradient step sees a slightly different sub-network;
# - dropped units contribute nothing on that step;
# - parameters attached only to dropped units are not updated on that step.
#
# At prediction time, we do **not** keep dropping units. Instead, Lindholm rescales the outgoing weights:
#
# $$
# \widetilde{W}^{(\ell)} = r W^{(\ell)}, \qquad (6.34a)
# $$
#
# $$
# q^{(\ell)} = h\left(\widetilde{W}^{(\ell)} q^{(\ell-1)} + b^{(\ell)}\right). \qquad (6.34b)
# $$
#
# Reason:
#
# - this keeps the expected input to the next layer consistent between training and testing.
#
# Dropout vs bagging:
#
# - bagging trains separate models with separate parameters;
# - dropout samples sub-networks that share parameters;
# - bagging commonly uses bootstrap datasets;
# - dropout typically uses ordinary mini-batches together with random masks.
#
# Why dropout can break or disappoint:
#
# - if the network is already too small, dropout can make optimisation unnecessarily hard;
# - if the keep probability is too low, the model may underfit;
# - if train/test scaling is mishandled, predictions become inconsistent.
#
# ### Objective, optimisation, failure modes
#
# The optimisation story does **not** change from Week 7:
#
# - define a loss, typically cross-entropy for classification;
# - run mini-batch gradient descent;
# - use backpropagation to compute the gradients.
#
# What changes in Week 8 is the architecture and the associated failure modes.
#
# Useful exam checklist:
#
# - **Objective:** classify images while exploiting local spatial structure.
# - **Optimisation:** train filters, dense weights, and biases using stochastic gradient methods and backpropagation.
# - **Failure modes:** output-shape mistakes, incorrect parameter counting, confusing pooling with convolution, forgetting that pooling has no trainable parameters, and overfitting when the network capacity is too high.
#
# ## Exam-oriented takeaways
#
# Most likely assessable Week 8 skills:
#
# 1. Explain why CNNs are better suited to images than fully connected networks.
# 2. Define sparse interactions and parameter sharing clearly.
# 3. Compute convolution outputs with and without stride/padding.
# 4. Compute max-pooling outputs and output sizes.
# 5. Count parameters in convolutional and dense parts of a CNN.
# 6. Explain dropout as an approximate ensemble / regularisation technique.
# 7. Distinguish what changes parameter count versus what only changes spatial dimensions.
#
# Fast exam checklist:
#
# - Can I explain in one sentence what parameter sharing buys us?
# - Can I compute output dimensions from filter size, padding, and stride?
# - Can I state why pooling adds no trainable parameters?
# - Can I explain why dropout needs train-time randomness but test-time weight scaling?
# - Can I count weights in a CNN layer without forgetting bias terms?
#
# ## Past exam questions (2023-2025)
#
# The exam index marks the following questions as Week 8-primary or clearly Week 8-relevant.
#
# ### 2023-B-6: convolution, stride, padding, and pooling
#
# Source: [2023 exam PDF](../../references/2023_COMP4702_exam.pdf), question id `2023-B-6`, extracted in [COMP4702_exams_2023_2025.md](../../references/COMP4702_exams_2023_2025.md).
#
# > (a) Given the following matrices: ... where `X` is the `5 × 5` input data (e.g. pixel values) and `K` is a `2 × 2` kernel, perform a convolution operation on `X` using `K` (stride of `1`).
# >
# > (b) Perform a convolution operation on `X` using `K` but use a stride of `2` and use padding (zero values): add a row to the bottom and a column to the right of `X` (making it a `6 × 6` matrix).
# >
# > (c) Perform `3 × 3` max pooling on matrix `X` (stride of `1`).
#
# Why it matters:
#
# - this is the most direct "do the arithmetic" Week 8 question;
# - it tests operation mechanics rather than verbal theory.
#
# ### 2024-A-7: conceptual CNN MCQ
#
# Source: [2024 exam PDF](../../references/2024_COMP4702_exam.pdf), question id `2024-A-7`.
#
# > Regarding Convolutional Neural Networks (CNNs), which of the following statement is incorrect?
# >
# > (a) CNNs use techniques such as stride and pooling to reduce the overall size of the network.
# >
# > (b) A CNN is only considered to be a deep neural network if the number of layers is greater than the number of hidden units in the largest layer.
# >
# > (c) CNNs enforce sparse interactions by only having incoming connections to a hidden unit from a small, localized region of the previous layer.
# >
# > (d) CNNs implement parameter sharing by having the same weights for all hidden units in a layer, using a filter that moves across all positions on the input.
#
# Why it matters:
#
# - this is pure definition-checking;
# - if the phrases "sparse interactions" and "parameter sharing" are not automatic for you, you will spend too long on easier marks.
#
# ### 2024-B-5(c): output-size arithmetic inside a CNN
#
# Source: [2024 exam PDF](../../references/2024_COMP4702_exam.pdf), question id `2024-B-5`, where Week 8 appears as a secondary week.
#
# > Consider a convolutional neural network where, at some point in the network:
# >
# > - the input data is a `6 × 6` matrix
# > - convolution is applied with a `3 × 3` kernel, followed by max pooling using a `2 × 2` kernel.
# >
# > Calculate the size of the output for these operations.
#
# Why it matters:
#
# - this is the cleanest reminder that output-shape questions can appear inside a broader neural-network question.
#
# ### 2025-B-8: parameter counting for dense versus convolutional networks
#
# Source: [2025 exam PDF](../../references/2025_COMP4702_exam.pdf), question id `2025-B-8`.
#
# > (a) Consider building a feed-forward neural network that uses this data as input and tries to classify each input as one of the printable characters. You decide to use two hidden layers with `50` units in each layer. Calculate the total number of weights in the network, including bias weights. Show your working.
# >
# > (b) Now consider using a convolutional neural network for this data. You decide that the network architecture will have:
# >
# > - The input layer.
# > - Convolutional layer 1, that has `6` filters with a filter size of `5 × 5`, zero padding of size `2` all around the bitmap and a stride of `2`.
# > - Convolutional layer 2, that has `4` filters with a filter size of `3 × 3`, no padding and a stride of `1`.
# > - A fully-connected layer of `10` hidden units.
# > - The output layer that uses softmax activation to produce probabilistic outputs.
# >
# > Calculate the total number of weights in the network, including bias weights. Show your working.
#
# Why it matters:
#
# - this combines the two most common Week 8 subskills: shape propagation and parameter counting.
#
# ## Toy example 1: convolution, stride, and max pooling by hand
#
# This mirrors the style of `2023-B-6` on a smaller input.

# %%
X = np.array(
    [
        [1, 0, 2, 1],
        [0, 1, 1, 0],
        [2, 1, 0, 1],
        [1, 0, 1, 2],
    ],
    dtype=float,
)

K = np.array(
    [
        [1, -1],
        [0, 1],
    ],
    dtype=float,
)

def conv2d_valid(x, k, stride=1):
    out_rows = (x.shape[0] - k.shape[0]) // stride + 1
    out_cols = (x.shape[1] - k.shape[1]) // stride + 1
    out = np.zeros((out_rows, out_cols), dtype=float)
    for i in range(out_rows):
        for j in range(out_cols):
            patch = x[i * stride:i * stride + k.shape[0], j * stride:j * stride + k.shape[1]]
            out[i, j] = np.sum(patch * k)
    return out

def max_pool2d(x, pool=2, stride=2):
    out_rows = (x.shape[0] - pool) // stride + 1
    out_cols = (x.shape[1] - pool) // stride + 1
    out = np.zeros((out_rows, out_cols), dtype=float)
    for i in range(out_rows):
        for j in range(out_cols):
            patch = x[i * stride:i * stride + pool, j * stride:j * stride + pool]
            out[i, j] = np.max(patch)
    return out

conv_stride_1 = conv2d_valid(X, K, stride=1)
conv_stride_2 = conv2d_valid(X, K, stride=2)
pool_after_conv = max_pool2d(conv_stride_1, pool=2, stride=2)

print("Input X:\n", X)
print("\nKernel K:\n", K)
print("\nConvolution (stride 1):\n", conv_stride_1)
print("\nConvolution (stride 2):\n", conv_stride_2)
print("\nMax pooling 2x2 with stride 2 after stride-1 convolution:\n", pool_after_conv)


# %% [markdown]
# #### Observations
#
# - The stride-1 convolution keeps more spatial detail because the filter is evaluated at more positions.
# - Increasing the stride reduces the output size immediately.
# - Pooling further condenses the output without adding parameters.
#
# #### Why this toy example matters
#
# - It builds the mechanics needed for `2023-B-6`.
# - It makes the distinction between a learned convolution operation and a parameter-free pooling operation very concrete.
#
# ## Toy example 2: CNN output shapes and parameter counts
#
# This is the most common Week 8 exam skill after raw convolution arithmetic.

# %%
def conv_output_size(n, f, p=0, s=1):
    return (n + 2 * p - f) // s + 1

# Example loosely aligned with 2025-B-8(b)
input_size = 8

conv1_size = conv_output_size(input_size, f=5, p=2, s=2)
conv2_size = conv_output_size(conv1_size, f=3, p=0, s=1)

conv1_params = (5 * 5 * 1 + 1) * 6      # grayscale input -> 6 filters
conv2_params = (3 * 3 * 6 + 1) * 4      # 6 input channels -> 4 filters
flattened_units = conv2_size * conv2_size * 4
dense_params = (flattened_units + 1) * 10
output_params = (10 + 1) * 95

total_params = conv1_params + conv2_params + dense_params + output_params

print(f"conv1 output size: {conv1_size} x {conv1_size}")
print(f"conv2 output size: {conv2_size} x {conv2_size}")
print(f"conv1 parameters: {conv1_params}")
print(f"conv2 parameters: {conv2_params}")
print(f"flattened units before dense layer: {flattened_units}")
print(f"dense-layer parameters: {dense_params}")
print(f"output-layer parameters: {output_params}")
print(f"total parameters: {total_params}")

# %% [markdown]
# #### Observations
#
# - The parameter count in a convolutional layer depends on filter size, input channels, and number of filters, not on the width/height of the whole image.
# - The dense layer often dominates parameter count once the convolutional features are flattened.
# - If you miscompute one output size early, every later count becomes wrong.
#
# #### Demo questions
#
# - Is your output what you expected? Does it match your understanding of the algorithms?
# - Is your model overfitting or underfitting?
# - What is the purpose of each task in the prac?
#
# ## Toy example 3: dropout masks and test-time scaling
#
# This shows the expectation-preserving idea behind Equation `(6.34)`.

# %%
q = np.array([2.0, -1.0, 3.0, 4.0])
r = 0.75

mask = (np.random.rand(q.size) < r).astype(float)
q_tilde = mask * q
test_time_scaled = r * q

print("Original activations:", q)
print("Sampled dropout mask:", mask)
print("Training-time masked activations:", q_tilde)
print("Expected activations used at test time:", test_time_scaled)

# %% [markdown]
# #### Observations
#
# - During training, a specific unit is either present or removed in a given sub-network.
# - Across many such samples, scaling by `r` at test time approximates the average contribution of those units.
# - This is why dropout behaves like a cheap approximate ensemble rather than a single fixed network during training.
#
# #### Demo questions
#
# - Is your output what you expected? Does it match your understanding of the algorithms?
# - Is your model overfitting or underfitting?
# - What is the purpose of each task in the prac?
#
# ## Week 8 wrap-up
#
# What to remember under time pressure:
#
# 1. CNNs are built for grid-structured data because they exploit local structure.
# 2. The two defining ideas are sparse interactions and parameter sharing.
# 3. Stride and pooling change output size; pooling does not add parameters.
# 4. Multiple filters create multiple channels, and later filters span all previous channels.
# 5. Dropout is a regularisation method that approximates an ensemble of shared-parameter sub-networks.
#
# Minimal revision script:
#
# - define convolutional layer in one sentence;
# - explain parameter sharing and sparse interactions;
# - do one by-hand output-size calculation;
# - do one parameter-count calculation with biases;
# - explain dropout train-time masking and test-time scaling.
#
# ## Sources used
#
# - [Course Summary Table v1 2026](../../references/CourseSummaryTable_v1_26.pdf)
# - [Lindholm (2022) main textbook](../../references/main-text-book-machine-learning-lindholm-2022.pdf), especially Chapter 6 sections `6.3` and `6.4`
# - [Exam index by week](../../references/exam_questions_2023_2025_by_week.csv)
# - [Extracted 2023-2025 exams](../../references/COMP4702_exams_2023_2025.md)
# - [Week 7 lecture notes](../week-07/week-07-lecture.md) for continuity of the chapter split
