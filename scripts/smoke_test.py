#!/usr/bin/env python3
"""Minimal environment smoke test for COMP4702."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def main() -> None:
    # Tiny toy dataset for a trivial binary classification fit.
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=int)

    # Touch pandas to verify import/runtime behavior.
    df = pd.DataFrame({"x": X.ravel(), "y": y})

    model = LogisticRegression(random_state=0, solver="lbfgs")
    model.fit(df[["x"]], df["y"])
    _ = model.predict(df[["x"]])

    # Touch matplotlib and display a simple plot.
    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Smoke Test Plot")
    plt.show()

    print("OK")


if __name__ == "__main__":
    main()
