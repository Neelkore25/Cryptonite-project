"""
cryptonite.data.simulator
--------------------------
Generates realistic-ish crypto account activity histories for:
  1) training the ML bot-classifier
  2) demoing the fuzzy trust engine
  3) tuning the fuzzy system with the genetic algorithm

In a production deployment, `fetch_account_history(user_id)` would be replaced
by real calls to an exchange / on-chain API (transactions, KYC status,
account creation date, device fingerprints, etc). Everything downstream
(feature extraction, fuzzy engine, ML model, GA tuning) is agnostic to where
the raw history came from.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

FEATURE_COLUMNS = [
    "account_age_days",       # how old the account is
    "txn_count",               # total number of transactions
    "txn_per_day",              # transaction frequency
    "avg_txn_interval_std",    # std-dev of time between txns (low = robotic/regular)
    "unique_counterparties",   # distinct wallets interacted with
    "counterparty_diversity",  # unique_counterparties / txn_count
    "avg_txn_value",
    "txn_value_variance",      # variance in transaction size (bots often near-identical)
    "night_activity_ratio",    # fraction of txns between 00:00-05:00 local
    "kyc_verified",            # 0/1
    "social_followers",
    "social_following",
    "follower_following_ratio",
    "flagged_reports",         # community/user reports against the account
    "withdrawal_to_deposit_ratio",
]

LABEL_COLUMN = "is_bot"


def _simulate_human(n):
    age = RNG.gamma(shape=4, scale=120, size=n) + 5          # organically old accounts
    txn_count = RNG.poisson(lam=age / 6) + 1
    txn_per_day = txn_count / np.maximum(age, 1)
    interval_std = RNG.normal(loc=18, scale=6, size=n).clip(2)   # irregular human timing
    unique_cp = (txn_count * RNG.uniform(0.4, 0.9, n)).astype(int) + 1
    cp_diversity = unique_cp / np.maximum(txn_count, 1)
    avg_val = RNG.lognormal(mean=4, sigma=1.3, size=n)
    val_var = avg_val * RNG.uniform(0.5, 3.0, n)
    night_ratio = RNG.beta(1.5, 6, n)
    kyc = RNG.binomial(1, 0.75, n)
    followers = RNG.negative_binomial(4, 0.02, n)
    following = RNG.negative_binomial(4, 0.05, n) + 1
    ff_ratio = followers / following
    reports = RNG.poisson(0.15, n)
    wd_dep = RNG.beta(2, 2, n) * 1.4

    return dict(
        account_age_days=age, txn_count=txn_count, txn_per_day=txn_per_day,
        avg_txn_interval_std=interval_std, unique_counterparties=unique_cp,
        counterparty_diversity=cp_diversity, avg_txn_value=avg_val,
        txn_value_variance=val_var, night_activity_ratio=night_ratio,
        kyc_verified=kyc, social_followers=followers, social_following=following,
        follower_following_ratio=ff_ratio, flagged_reports=reports,
        withdrawal_to_deposit_ratio=wd_dep,
    )


def _simulate_bot(n):
    age = RNG.gamma(shape=1.3, scale=25, size=n) + 1          # young accounts
    txn_per_day = RNG.gamma(shape=6, scale=6, size=n) + 1       # bursty high frequency
    txn_count = (txn_per_day * age).astype(int) + 1
    interval_std = RNG.normal(loc=2.0, scale=1.2, size=n).clip(0.05)  # robotic regularity
    unique_cp = (txn_count * RNG.uniform(0.02, 0.15, n)).astype(int) + 1  # low diversity
    cp_diversity = unique_cp / np.maximum(txn_count, 1)
    avg_val = RNG.lognormal(mean=3, sigma=0.3, size=n)          # near-identical values
    val_var = avg_val * RNG.uniform(0.01, 0.2, n)
    night_ratio = RNG.beta(5, 2, n)                              # runs while humans sleep
    kyc = RNG.binomial(1, 0.08, n)
    followers = RNG.negative_binomial(2, 0.2, n)
    following = RNG.negative_binomial(20, 0.05, n) + 1           # follows many, few follow back
    ff_ratio = followers / following
    reports = RNG.poisson(2.2, n)
    wd_dep = RNG.beta(5, 1, n) * 1.4                              # drains funds out

    return dict(
        account_age_days=age, txn_count=txn_count, txn_per_day=txn_per_day,
        avg_txn_interval_std=interval_std, unique_counterparties=unique_cp,
        counterparty_diversity=cp_diversity, avg_txn_value=avg_val,
        txn_value_variance=val_var, night_activity_ratio=night_ratio,
        kyc_verified=kyc, social_followers=followers, social_following=following,
        follower_following_ratio=ff_ratio, flagged_reports=reports,
        withdrawal_to_deposit_ratio=wd_dep,
    )


def generate_dataset(n_humans=1500, n_bots=1500, seed=42, noise=0.25):
    """
    noise: fraction of accounts that are "edge cases" -- sophisticated bots
    that mimic human patterns (or inactive humans that look bot-like).
    This keeps the classification task realistically imperfect instead of
    trivially separable, matching what real exchange data looks like.
    """
    global RNG
    RNG = np.random.default_rng(seed)
    humans = pd.DataFrame(_simulate_human(n_humans))
    humans[LABEL_COLUMN] = 0
    bots = pd.DataFrame(_simulate_bot(n_bots))
    bots[LABEL_COLUMN] = 1
    df = pd.concat([humans, bots], ignore_index=True)
    for col in FEATURE_COLUMNS:
        if col != "kyc_verified":
            df[col] = df[col].astype(float)

    # Blend a slice of each class toward the other class's centroid to create
    # realistic overlap (sophisticated bots / dormant humans).
    n_noisy = int(len(df) * noise)
    noisy_idx = RNG.choice(df.index, size=n_noisy, replace=False)
    numeric_cols = [c for c in FEATURE_COLUMNS if c != "kyc_verified"]
    for idx in noisy_idx:
        is_bot = df.loc[idx, LABEL_COLUMN] == 1
        template = _simulate_human(1) if is_bot else _simulate_bot(1)
        blend = RNG.uniform(0.35, 0.65)
        for col in numeric_cols:
            df.loc[idx, col] = (1 - blend) * df.loc[idx, col] + blend * template[col][0]

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df[FEATURE_COLUMNS + [LABEL_COLUMN]]


# Bundled demo accounts (used by main.py --demo and the dashboard's "live
# watchlist") are labelled EXPLICITLY here so the precomputed batch report
# and the dashboard's in-browser re-analysis of the same ids always agree.
# Any id NOT in this map falls back to a deterministic hash + keyword
# heuristic, purely for demo variety on ad-hoc user-typed ids.
DEMO_LABELS = {
    "alice_trader_2019": "human",
    "quickflip_bot_007": "bot",
    "0xAnon_new_wallet": "bot",
    "grandma_hodler": "human",
    "airdrop_hunter_x1000": "bot",
    "satoshi_disciple": "human",
    "pump_signal_caller": "bot",
    "long_term_hodler_99": "human",
}


def _stable_hash(s: str) -> int:
    """FNV-1a 32-bit hash -- deterministic across processes/languages
    (unlike Python's built-in hash(), which is salted per-process)."""
    h = 0x811c9dc5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def fetch_account_history(user_id: str, seed=None):
    """
    Stand-in for a real API call. Deterministically simulates ONE account's
    feature vector from a user_id string so demo runs are repeatable.
    Replace this function with a real exchange/on-chain data connector.
    """
    if user_id in DEMO_LABELS:
        is_bot_like = DEMO_LABELS[user_id] == "bot"
    else:
        h_check = user_id.lower()
        keyword_hint = any(k in h_check for k in
                            ["bot", "quickflip", "airdrop", "free", "x1000", "pump"])
        hash_hint = (_stable_hash(user_id) % 3 == 0)
        is_bot_like = keyword_hint or hash_hint

    h = _stable_hash(user_id)
    global RNG
    RNG = np.random.default_rng(h)
    row = _simulate_bot(1) if is_bot_like else _simulate_human(1)
    row = {k: v[0] for k, v in row.items()}
    return row


if __name__ == "__main__":
    df = generate_dataset()
    print(df.describe())
    df.to_csv("/home/claude/cryptonite/data/synthetic_accounts.csv", index=False)
    print("Saved synthetic_accounts.csv", df.shape)
