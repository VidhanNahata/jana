"""
Pure numpy / scikit-learn part of the CLIP option (no torch import, so it is unit-testable).

A pretrained CLIP image encoder turns a photo into a 512-number "fingerprint". We L2-normalise it
and fit a small multinomial logistic regression on top (the "head"). Only the head is trained:
seconds on a CPU, and it works with ~100 photos per class.
"""

import numpy as np

from common import softmax


def l2norm(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9, None)


def fit_head(X, y, C=1.0, seed=0):
    """Return (W[n_classes, d], b[n_classes]) for softmax(X @ W.T + b). Classes are balanced by weight."""
    from sklearn.linear_model import LogisticRegression
    n = int(y.max()) + 1
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(X, y)
    W = np.zeros((n, X.shape[1]), np.float32)
    b = np.full(n, -20.0, np.float32)            # a class absent from training can never win
    for row, cls in enumerate(clf.classes_):
        W[cls] = clf.coef_[row] if len(clf.classes_) > 2 else (clf.coef_[0] if row == 1 else -clf.coef_[0])
        b[cls] = clf.intercept_[row] if len(clf.classes_) > 2 else (clf.intercept_[0] if row == 1 else -clf.intercept_[0])
    return W, b


def head_probs(X, W, b):
    return softmax(X @ W.T + b)


def zero_shot_probs(X, text_emb, scale=100.0):
    """CLIP's own text-matching: softmax over cosine similarity to one prompt per class."""
    return softmax(scale * X @ l2norm(text_emb).T)


def macro_f1(y_true, y_pred, n):
    f1s = []
    for c in range(n):
        tp = np.sum((y_true == c) & (y_pred == c)); fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)), f1s


def choose_C(Xtr, ytr, Xva, yva, grid=(0.1, 0.3, 1, 3, 10, 30, 100)):
    """Regularisation strength with the best validation macro-F1 (stronger regularisation = safer on tiny data)."""
    n = int(max(ytr.max(), yva.max())) + 1
    best = None
    for C in grid:
        W, b = fit_head(Xtr, ytr, C)
        f1, _ = macro_f1(yva, head_probs(Xva, W, b).argmax(1), n)
        if best is None or f1 > best[0] + 1e-9:
            best = (f1, C)
    return best[1], best[0]
