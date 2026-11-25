"""Confidence scoring, calibration, and classification.

This module sits **between** the raw model outputs and the values
shown to the user.  It does three things:

1. **Classify** a raw score into high / medium / low using the
   configurable thresholds from ``config.py``.

2. **Calibrate** raw softmax scores using temperature scaling so
   they better approximate true probabilities.  Temperature
   scaling learns a single scalar *T* on a held-out validation set
   that minimises the negative log-likelihood; see Guo et al.
   "On Calibration of Modern Neural Networks" (ICML 2017).

3. **Evaluate** calibration quality with Expected Calibration Error
   (ECE) and Brier score.

Important
---------
* Until ``fit()`` has been called on real validation data the
  calibrator operates as an identity (T = 1.0), i.e. it passes
  raw scores through unchanged.
* This module does **not** invent calibration numbers.  Calibration
  metrics are only meaningful after ``fit()`` is called with actual
  (prediction, ground-truth) pairs.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from app.config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM

logger = logging.getLogger(__name__)


# ── Confidence classification ────────────────────────────────────────

def classify_confidence(score: float) -> str:
    """Return ``'high'``, ``'medium'``, or ``'low'``."""
    if score >= CONFIDENCE_HIGH:
        return "high"
    if score >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def field_confidence_detail(score: float, calibrated: bool = False) -> dict:
    """Convenience wrapper returning a JSON-friendly dict."""
    return {
        "score": round(score, 4),
        "level": classify_confidence(score),
        "calibrated": calibrated,
    }


# ── Temperature-scaling calibrator ───────────────────────────────────

class TemperatureCalibrator:
    """Post-hoc temperature scaling for confidence calibration.

    After training (or loading a pretrained model), call ``fit()``
    with validation predictions and ground-truth labels to learn the
    optimal temperature.  Then ``calibrate()`` rescales new scores.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 200,
    ) -> float:
        """Learn temperature via gradient descent on NLL.

        Parameters
        ----------
        logits : ndarray, shape (N, C)
            Raw logits (pre-softmax) from the model.
        labels : ndarray, shape (N,)
            Integer ground-truth class indices.
        lr : float
            Learning rate for gradient descent.
        max_iter : int
            Maximum optimisation steps.

        Returns
        -------
        float
            Learned temperature.
        """
        T = 1.0
        N = len(labels)
        if N == 0:
            logger.warning("fit() called with empty data; T stays 1.0")
            return T

        for step in range(max_iter):
            scaled = logits / T
            # Numerically stable softmax
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            exp_shifted = np.exp(shifted)
            probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

            # NLL
            nll = -np.log(probs[np.arange(N), labels] + 1e-12).mean()

            # Gradient of NLL w.r.t. T
            grad = 0.0
            for i in range(N):
                p = probs[i]
                y = labels[i]
                weighted = (p * logits[i]).sum()
                grad += (weighted - logits[i, y]) / (T * T)
            grad /= N

            T -= lr * grad
            T = max(T, 0.01)  # prevent collapse

        self.temperature = T
        self._fitted = True
        logger.info(
            "Temperature calibration: T=%.4f  (NLL=%.4f, N=%d)",
            T,
            nll,
            N,
        )
        return T

    def calibrate(self, score: float) -> float:
        """Apply temperature scaling to a single confidence score.

        For a single-class softmax score, temperature scaling maps:
            calibrated = sigmoid(logit(score) / T)
        where logit(p) = log(p / (1-p)).
        """
        if not self._fitted or self.temperature == 1.0:
            return score
        # Clamp to avoid log(0)
        p = max(min(score, 1.0 - 1e-7), 1e-7)
        logit = math.log(p / (1.0 - p))
        scaled = logit / self.temperature
        return 1.0 / (1.0 + math.exp(-scaled))

    def save(self, path: str | Path) -> None:
        """Persist calibration parameters to a JSON file."""
        Path(path).write_text(
            json.dumps(
                {"temperature": self.temperature, "fitted": self._fitted}
            )
        )

    def load(self, path: str | Path) -> None:
        """Load previously saved calibration parameters."""
        p = Path(path)
        if not p.is_file():
            logger.warning("Calibration file not found: %s", p)
            return
        data = json.loads(p.read_text())
        self.temperature = data.get("temperature", 1.0)
        self._fitted = data.get("fitted", False)
        logger.info("Loaded calibration: T=%.4f", self.temperature)


# ── Calibration evaluation metrics ───────────────────────────────────

def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute ECE and per-bin statistics for a reliability diagram.

    Parameters
    ----------
    confidences : ndarray, shape (N,)
        Predicted confidence for each sample (in [0, 1]).
    accuracies : ndarray, shape (N,)
        Binary correctness (1 = correct, 0 = incorrect).
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    dict
        ``ece`` : float — expected calibration error.
        ``bins`` : list of dicts with ``lower``, ``upper``,
        ``avg_confidence``, ``avg_accuracy``, ``count``.
    """
    bins_info = []
    ece = 0.0
    total = len(confidences)
    if total == 0:
        return {"ece": 0.0, "bins": [], "n_samples": 0}

    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            bins_info.append(
                {
                    "lower": round(float(lo), 2),
                    "upper": round(float(hi), 2),
                    "avg_confidence": None,
                    "avg_accuracy": None,
                    "count": 0,
                }
            )
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc = float(accuracies[mask].mean())
        ece += (count / total) * abs(avg_acc - avg_conf)
        bins_info.append(
            {
                "lower": round(float(lo), 2),
                "upper": round(float(hi), 2),
                "avg_confidence": round(avg_conf, 4),
                "avg_accuracy": round(avg_acc, 4),
                "count": count,
            }
        )

    return {
        "ece": round(ece, 4),
        "bins": bins_info,
        "n_samples": total,
    }


def brier_score(confidences: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted confidence and outcome."""
    return float(np.mean((confidences - outcomes) ** 2))
