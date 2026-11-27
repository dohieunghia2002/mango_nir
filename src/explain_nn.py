"""Attributions for the 1D-CNN (Step 4b): Integrated Gradients (IG) and gradient x input (GxI).

The attribution functions take a gradient function grad_fn(X) -> dF/dX (one row per spectrum), so their numerics are
tested locally on functions with known gradients; tf_gradient_fn / tf_predict_fn build these functions for a Keras model
(TensorFlow is imported lazily, on Colab).
IG with baseline b: (x - b) * mean over alpha of grad(b + alpha (x - b)), midpoint rule with `steps` points; the
attributions then sum to F(x) - F(b) up to the integration error (completeness), which is reported.
"""
import numpy as np


def integrated_gradients(grad_fn, X, baseline, steps=64, batch=32):
    X = np.asarray(X, dtype=np.float32)
    base = np.broadcast_to(np.asarray(baseline, dtype=np.float32), X.shape[1:])
    alphas = ((np.arange(steps) + 0.5) / steps).astype(np.float32)
    out = np.empty(X.shape, dtype=np.float64)
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        d = xb - base
        path = base + alphas[:, None, None] * d[None]                       # steps x b x p
        g = np.asarray(grad_fn(path.reshape(-1, X.shape[1])), dtype=np.float64).reshape(steps, len(xb), X.shape[1])
        out[i:i + batch] = g.mean(axis=0) * d
    return out


def gradient_x_input(grad_fn, X, baseline):
    X = np.asarray(X, dtype=np.float32)
    return np.asarray(grad_fn(X), dtype=np.float64) * (X - np.asarray(baseline, dtype=np.float32))


def completeness_error(attr, f_x, f_base):
    """|sum of attributions - (F(x) - F(baseline))| per row, and relative to |F(x) - F(baseline)|."""
    gap = np.asarray(f_x, dtype=np.float64) - np.asarray(f_base, dtype=np.float64)
    err = np.abs(attr.sum(axis=1) - gap)
    return err, err / np.maximum(np.abs(gap), 1e-8)


def tf_predict_fn(model, batch=8192):
    """Fast inference: direct calls of a compiled tf.function instead of model.predict."""
    import tensorflow as tf

    @tf.function(reduce_retracing=True)
    def f(x):
        return model(x, training=False)

    def predict(X):
        X = np.asarray(X, dtype=np.float32)
        return np.concatenate([f(tf.constant(X[i:i + batch])).numpy().ravel() for i in range(0, len(X), batch)])
    return predict


def tf_gradient_fn(model, batch=8192):
    import tensorflow as tf

    @tf.function(reduce_retracing=True)
    def g(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x, training=False)
        return tape.gradient(y, x)    # gradient of the sum over rows = per-row gradients (rows are independent)

    def grad(X):
        X = np.asarray(X, dtype=np.float32)
        return np.concatenate([g(tf.constant(X[i:i + batch])).numpy() for i in range(0, len(X), batch)])
    return grad
