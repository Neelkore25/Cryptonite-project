"""
cryptonite.engine.feature_engineering
---------------------------------------
Converts a raw account-history feature dict (as produced by
data.simulator.fetch_account_history, or by a real exchange API) into:

  (a) the 4 normalized [0,100] inputs the Fuzzy Trust Engine expects
  (b) the full numeric feature vector the ML Bot Detector expects

Keeping this in one place means the fuzzy engine and the ML model always
see a consistent, documented transformation of the same raw signals.
"""

import numpy as np
from cryptonite.data.simulator import FEATURE_COLUMNS


def _norm(x, lo, hi):
    """Clip-and-scale x from [lo, hi] to [0, 100]."""
    if hi == lo:
        return 50.0
    val = (x - lo) / (hi - lo) * 100
    return float(np.clip(val, 0, 100))


def to_fuzzy_inputs(raw: dict) -> dict:
    """Map raw account features -> the 4 linguistic-variable crisp inputs."""

    # account_age: log-scaled, 0 days -> 0, ~2 years+ -> 100
    account_age = _norm(np.log1p(raw["account_age_days"]), np.log1p(1), np.log1p(730))

    # txn_regularity: humans have IRREGULAR timing + varied values (higher = more natural/human).
    # We combine interval-std (higher=more natural) and value-variance-ratio.
    interval_component = _norm(raw["avg_txn_interval_std"], 0, 25)
    value_var_ratio = raw["txn_value_variance"] / max(raw["avg_txn_value"], 1e-6)
    value_component = _norm(value_var_ratio, 0, 3)
    txn_regularity = 0.6 * interval_component + 0.4 * value_component

    # network_diversity: counterparty diversity + inverse night-activity skew
    diversity_component = _norm(raw["counterparty_diversity"], 0, 0.8)
    night_penalty = _norm(1 - raw["night_activity_ratio"], 0, 1)
    network_diversity = 0.7 * diversity_component + 0.3 * night_penalty

    # verification: KYC + healthy social signal - report penalty
    kyc_component = raw["kyc_verified"] * 100
    ff_component = _norm(raw["follower_following_ratio"], 0, 2)
    report_penalty = _norm(5 - raw["flagged_reports"], 0, 5)
    verification = 0.5 * kyc_component + 0.25 * ff_component + 0.25 * report_penalty

    return dict(
        account_age=account_age,
        txn_regularity=txn_regularity,
        network_diversity=network_diversity,
        verification=verification,
    )


def to_ml_vector(raw: dict) -> np.ndarray:
    """Return the ordered numeric feature vector the ML model was trained on."""
    return np.array([raw[c] for c in FEATURE_COLUMNS], dtype=float)
