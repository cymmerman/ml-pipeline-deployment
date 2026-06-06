"""
drift.py — Feature Drift Detection
Kolmogorov-Smirnov test comparing live sensor readings to training distribution.
Flags when incoming data has shifted outside the model's training domain.
Critical for aerospace applications where sensor calibration drifts over time.
"""

import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Optional

logger = logging.getLogger(__name__)

# Path to saved training statistics
DRIFT_STATS_PATH = Path(__file__).parent / "model" / "drift_stats.json"

# KS-test p-value threshold — below this means drift is detected
# 0.05 = 95% confidence that distributions differ
KS_THRESHOLD = 0.05

# Severity thresholds on KS statistic (0-1 scale)
SEVERITY_MODERATE = 0.3
SEVERITY_HIGH     = 0.6


# ── Training stats ────────────────────────────────────────────────────────────
class DriftBaseline:
    """
    Holds per-feature statistics computed from training data.
    Loaded once at startup from drift_stats.json.
    If the file doesn't exist, it's computed from the scaler parameters.
    """
    feature_names: list = []
    means: np.ndarray  = None
    stds: np.ndarray   = None
    mins: np.ndarray   = None
    maxs: np.ndarray   = None
    percentiles: dict  = {}   # {feature_name: [p5, p25, p50, p75, p95]}
    training_samples: list = []   # sample of scaled training vectors for KS test

    @classmethod
    def load(cls, scaler=None, feature_names: list = None):
        if DRIFT_STATS_PATH.exists():
            with open(DRIFT_STATS_PATH) as f:
                data = json.load(f)
            cls.feature_names  = data["feature_names"]
            cls.means          = np.array(data["means"])
            cls.stds           = np.array(data["stds"])
            cls.mins           = np.array(data["mins"])
            cls.maxs           = np.array(data["maxs"])
            cls.percentiles    = data["percentiles"]
            cls.training_samples = np.array(data["training_samples"])
            logger.info(f"Drift baseline loaded from {DRIFT_STATS_PATH}")
        elif scaler is not None:
            cls._build_from_scaler(scaler, feature_names or [])
        else:
            logger.warning("No drift baseline found and no scaler provided — drift detection disabled.")

    @classmethod
    def _build_from_scaler(cls, scaler, feature_names: list):
        """
        Reconstruct training distribution from scaler parameters.
        MinMaxScaler stores data_min_, data_max_, mean_ etc.
        We synthesize a reference distribution for KS testing.
        """
        logger.info("Building drift baseline from scaler parameters...")
        cls.feature_names = feature_names

        if hasattr(scaler, "data_min_"):
            cls.mins  = scaler.data_min_
            cls.maxs  = scaler.data_max_
            cls.means = (scaler.data_min_ + scaler.data_max_) / 2
            cls.stds  = (scaler.data_max_ - scaler.data_min_) / 4
        elif hasattr(scaler, "mean_"):
            cls.means = scaler.mean_
            cls.stds  = np.sqrt(scaler.var_)
            cls.mins  = cls.means - 3 * cls.stds
            cls.maxs  = cls.means + 3 * cls.stds
        else:
            logger.warning("Unrecognized scaler type — using unit normal baseline.")
            n = len(feature_names)
            cls.means = np.zeros(n)
            cls.stds  = np.ones(n)
            cls.mins  = np.full(n, -3.0)
            cls.maxs  = np.full(n,  3.0)

        # Synthesize reference samples using uniform distribution between min/max
        # This gives us a proper empirical distribution for KS testing
        rng = np.random.default_rng(42)
        n_samples = 500
        cls.training_samples = rng.uniform(
            low=cls.mins, high=cls.maxs, size=(n_samples, len(cls.means))
        )

        # Compute percentiles
        cls.percentiles = {}
        for i, name in enumerate(cls.feature_names):
            col = cls.training_samples[:, i]
            cls.percentiles[name] = {
                "p5":  float(np.percentile(col, 5)),
                "p25": float(np.percentile(col, 25)),
                "p50": float(np.percentile(col, 50)),
                "p75": float(np.percentile(col, 75)),
                "p95": float(np.percentile(col, 95)),
            }

        # Persist for future startups
        cls._save()
        logger.info("Drift baseline built and saved.")

    @classmethod
    def _save(cls):
        data = {
            "feature_names":    cls.feature_names,
            "means":            cls.means.tolist(),
            "stds":             cls.stds.tolist(),
            "mins":             cls.mins.tolist(),
            "maxs":             cls.maxs.tolist(),
            "percentiles":      cls.percentiles,
            "training_samples": cls.training_samples.tolist(),
        }
        with open(DRIFT_STATS_PATH, "w") as f:
            json.dump(data, f)

    @classmethod
    def is_ready(cls) -> bool:
        return cls.training_samples is not None and len(cls.training_samples) > 0


# ── Core drift detection ──────────────────────────────────────────────────────
def compute_drift(
    live_samples: np.ndarray,
    feature_names: Optional[list] = None,
) -> dict:
    """
    Run per-feature KS test comparing live_samples to training distribution.

    Args:
        live_samples: np.ndarray of shape (n_samples, n_features)
                      Should be RAW (unscaled) sensor values.
        feature_names: optional override for feature names

    Returns:
        Full drift report dict.
    """
    if not DriftBaseline.is_ready():
        return {"error": "Drift baseline not initialized."}

    names = feature_names or DriftBaseline.feature_names
    n_features = live_samples.shape[1] if live_samples.ndim == 2 else len(live_samples)

    feature_reports = []
    drifted = []

    for i in range(min(n_features, len(DriftBaseline.training_samples[0]))):
        name = names[i] if i < len(names) else f"feature_{i}"
        live_col     = live_samples[:, i] if live_samples.ndim == 2 else np.array([live_samples[i]])
        training_col = DriftBaseline.training_samples[:, i]

        # KS test — compares empirical CDFs
        ks_stat, p_value = stats.ks_2samp(training_col, live_col)

        drift_detected = p_value < KS_THRESHOLD

        # Severity based on KS statistic magnitude
        if ks_stat >= SEVERITY_HIGH:
            severity = "high"
        elif ks_stat >= SEVERITY_MODERATE:
            severity = "moderate"
        else:
            severity = "low"

        # Z-score of live mean vs training mean
        live_mean = float(np.mean(live_col))
        z_score   = float((live_mean - DriftBaseline.means[i]) / (DriftBaseline.stds[i] + 1e-8))

        report = {
            "feature":        name,
            "drift_detected": drift_detected,
            "severity":       severity if drift_detected else "none",
            "ks_statistic":   round(float(ks_stat), 6),
            "p_value":        round(float(p_value), 6),
            "z_score":        round(z_score, 4),
            "live_mean":      round(live_mean, 6),
            "training_mean":  round(float(DriftBaseline.means[i]), 6),
            "training_p5":    DriftBaseline.percentiles.get(name, {}).get("p5"),
            "training_p95":   DriftBaseline.percentiles.get(name, {}).get("p95"),
            "in_range":       bool(
                DriftBaseline.mins[i] <= live_mean <= DriftBaseline.maxs[i]
            ),
        }
        feature_reports.append(report)
        if drift_detected:
            drifted.append(name)

    overall_drift = len(drifted) > 0
    high_severity = [r["feature"] for r in feature_reports if r["severity"] == "high"]

    return {
        "drift_detected":    overall_drift,
        "drifted_features":  drifted,
        "drifted_count":     len(drifted),
        "high_severity":     high_severity,
        "overall_severity":  _overall_severity(feature_reports),
        "recommendation":    _recommendation(drifted, high_severity),
        "feature_reports":   feature_reports,
        "n_live_samples":    int(live_samples.shape[0]) if live_samples.ndim == 2 else 1,
        "ks_threshold":      KS_THRESHOLD,
    }


def compute_single_drift(features: list, feature_names: Optional[list] = None) -> dict:
    """Drift check for a single sample — wraps as a 1-row array."""
    arr = np.array(features).reshape(1, -1)
    return compute_drift(arr, feature_names)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _overall_severity(reports: list) -> str:
    severities = [r["severity"] for r in reports]
    if "high" in severities:
        return "high"
    if "moderate" in severities:
        return "moderate"
    if "low" in severities:
        return "low"
    return "none"


def _recommendation(drifted: list, high_severity: list) -> str:
    if not drifted:
        return "No drift detected. Sensor readings are within training distribution."
    if high_severity:
        return (
            f"HIGH severity drift detected on {', '.join(high_severity)}. "
            "Recommend sensor calibration check and model revalidation before "
            "relying on predictions. Consider retraining with recent data."
        )
    return (
        f"Moderate drift detected on {', '.join(drifted)}. "
        "Monitor closely. If drift persists over multiple readings, "
        "consider scheduling sensor inspection."
    )