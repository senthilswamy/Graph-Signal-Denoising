import pandas as pd
import numpy as np
import tensorflow as tf
# ==========================================================
# FILE: data_loader.py
# ==========================================================
#
# Purpose:
# Handles dataset loading and preprocessing.
#
# Operations:
# 1. Read traffic dataset
# 2. Missing value replacement
# 3. Z-score normalization
# 4. Graph adjacency matrix construction
# 5. Graph normalization
# 6. Input tensor creation
#
# Output:
# X             -> Model input
# A_hat         -> Normalized graph matrix
# num_nodes     -> Number of sensors
# numeric_cols  -> Original feature names
#
# ==========================================================

def load_dataset(path):

    df = pd.read_csv(path)

    df = df.fillna(df.median(numeric_only=True))

    numeric_cols = df.select_dtypes(include=np.number).columns

    mu = df[numeric_cols].mean()
    sigma = df[numeric_cols].std()

    df[numeric_cols] = (
        (df[numeric_cols] - mu) / sigma
    )

    data = df[numeric_cols].values.astype(np.float32)

    num_nodes = data.shape[1]

    avg_flow = np.mean(data, axis=0)

    gamma = 1.0

    W = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):

            W[i, j] = np.exp(
                -((avg_flow[i] - avg_flow[j]) ** 2)
                / gamma**2
            )

    A = W + np.eye(num_nodes)

    D = np.diag(np.sum(A, axis=1))

    D_inv_sqrt = np.linalg.inv(np.sqrt(D))

    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    A_hat = tf.constant(
        A_hat,
        dtype=tf.float32
    )

    X = np.expand_dims(data, axis=-1)

    return X, A_hat, num_nodes, numeric_cols