"""
cryptonite.data.etherscan_connector
--------------------------------------
A REAL data connector -- fetches an Ethereum wallet's actual transaction
history from the Etherscan API and converts it into the same feature-dict
shape that data.simulator.fetch_account_history() produces, so it drops
straight into the existing fuzzy + ANN + alert pipeline with zero changes
to engine/*.

Get a free API key at https://etherscan.io/apis (no approval wait, just an
email signup). Public on-chain data only -- no user login or consent flow
needed, which is why this is the easiest real source to wire up first.

Usage:
    export ETHERSCAN_API_KEY=your_key_here
    python3 -m cryptonite.data.etherscan_connector 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045

Or from code:
    from cryptonite.data.etherscan_connector import fetch_onchain_history
    raw = fetch_onchain_history("0x...", api_key="...")
"""

import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Explicitly load the .env file that sits next to this project (cryptonite/.env)
# rather than relying on python-dotenv's automatic directory-guessing, which
# behaves inconsistently depending on how/where the script is launched from.
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_ENV_PATH)
except ImportError:
    _ENV_PATH = None  # python-dotenv not installed -- env vars set the normal way still work

ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"
ETH_MAINNET_CHAIN_ID = 1  # Etherscan API V2 is multichain; 1 = Ethereum mainnet.
# Other common chain ids if you want to extend this later: 8453 Base, 137 Polygon,
# 56 BNB Chain, 42161 Arbitrum One -- see https://docs.etherscan.io/v2-migration

# Signals Etherscan simply doesn't expose (KYC status, social graph, community
# reports live outside the chain). We default them to neutral values rather
# than zero, so an otherwise-clean wallet isn't unfairly penalized just for
# lacking off-chain data the API can't provide. This is the main honest
# limitation of an on-chain-only connector -- see README.
UNAVAILABLE_DEFAULTS = dict(
    kyc_verified=0,              # not observable on-chain
    social_followers=50,         # neutral placeholder
    social_following=50,         # neutral placeholder
    flagged_reports=0,           # would come from your own report DB
)


class EtherscanError(Exception):
    pass


def _get(params, api_key, timeout=15):
    params = {**params, "apikey": api_key, "chainid": ETH_MAINNET_CHAIN_ID}
    resp = requests.get(ETHERSCAN_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # Etherscan returns status "0" both for genuine errors AND for the
    # (non-error) case "no transactions found" -- distinguish by message.
    if data.get("status") == "0" and data.get("message") != "No transactions found":
        raise EtherscanError(data.get("result", "Unknown Etherscan API error"))
    return data.get("result", [])


def fetch_raw_transactions(address: str, api_key: str, max_txns: int = 2000):
    """Pull normal + internal transactions for an address (paginated)."""
    txns = []
    page, offset = 1, min(max_txns, 1000)
    while len(txns) < max_txns:
        batch = _get({
            "module": "account", "action": "txlist", "address": address,
            "startblock": 0, "endblock": 99999999, "page": page,
            "offset": offset, "sort": "asc",
        }, api_key)
        if not batch:
            break
        txns.extend(batch)
        if len(batch) < offset:
            break
        page += 1
        time.sleep(0.25)  # stay under Etherscan's free-tier rate limit
    return txns[:max_txns]


def _features_from_transactions(address: str, txns: list) -> dict:
    address = address.lower()
    if not txns:
        raise EtherscanError(f"No on-chain transactions found for {address}")

    timestamps = sorted(int(t["timeStamp"]) for t in txns)
    now = time.time()
    account_age_days = max((now - timestamps[0]) / 86400, 0.1)
    txn_count = len(txns)
    txn_per_day = txn_count / account_age_days

    # Regularity: std-dev of gaps between consecutive transactions (hours).
    # Very low std => suspiciously robotic timing.
    gaps_hours = [(timestamps[i+1] - timestamps[i]) / 3600 for i in range(len(timestamps) - 1)]
    avg_txn_interval_std = statistics.pstdev(gaps_hours) if len(gaps_hours) > 1 else 12.0

    counterparties = set()
    values_eth = []
    deposit_total, withdrawal_total = 0.0, 0.0
    night_count = 0

    for t in txns:
        frm, to = t.get("from", "").lower(), t.get("to", "").lower()
        other = to if frm == address else frm
        if other:
            counterparties.add(other)

        val_eth = int(t.get("value", 0)) / 1e18
        values_eth.append(val_eth)

        if frm == address:
            withdrawal_total += val_eth
        elif to == address:
            deposit_total += val_eth

        hour_utc = datetime.fromtimestamp(int(t["timeStamp"]), tz=timezone.utc).hour
        if 0 <= hour_utc < 5:
            night_count += 1

    unique_counterparties = len(counterparties)
    counterparty_diversity = unique_counterparties / txn_count
    avg_txn_value = statistics.mean(values_eth) if values_eth else 0.0
    txn_value_variance = statistics.pvariance(values_eth) if len(values_eth) > 1 else 0.0
    night_activity_ratio = night_count / txn_count
    withdrawal_to_deposit_ratio = withdrawal_total / max(deposit_total, 1e-6)
    withdrawal_to_deposit_ratio = min(withdrawal_to_deposit_ratio, 5.0)  # clip outliers

    raw = dict(
        account_age_days=account_age_days,
        txn_count=txn_count,
        txn_per_day=txn_per_day,
        avg_txn_interval_std=avg_txn_interval_std,
        unique_counterparties=unique_counterparties,
        counterparty_diversity=counterparty_diversity,
        avg_txn_value=avg_txn_value,
        txn_value_variance=txn_value_variance,
        night_activity_ratio=night_activity_ratio,
        withdrawal_to_deposit_ratio=withdrawal_to_deposit_ratio,
    )
    raw["follower_following_ratio"] = (
        UNAVAILABLE_DEFAULTS["social_followers"] / UNAVAILABLE_DEFAULTS["social_following"]
    )
    raw.update({k: v for k, v in UNAVAILABLE_DEFAULTS.items() if k != "follower_following_ratio"})
    return raw


def fetch_onchain_history(address: str, api_key: str = None, max_txns: int = 2000) -> dict:
    """
    Real replacement for data.simulator.fetch_account_history(), backed by
    the Etherscan API. Returns the same feature-dict shape.
    """
    api_key = api_key or os.environ.get("ETHERSCAN_API_KEY")
    if not api_key:
        where = f" (looked for it at: {_ENV_PATH})" if _ENV_PATH else ""
        exists = f" -- file exists: {_ENV_PATH.exists()}" if _ENV_PATH else ""
        raise EtherscanError(
            "No Etherscan API key found. Get a free one at https://etherscan.io/apis "
            "and either pass api_key=... or set the ETHERSCAN_API_KEY env var.\n"
            f"Checked for a .env file{where}{exists}. "
            "If you just created/edited .env, make sure you RESTARTED the server "
            "(the .env file is only read once, at startup)."
        )
    txns = fetch_raw_transactions(address, api_key, max_txns=max_txns)
    return _features_from_transactions(address, txns)


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python3 -m cryptonite.data.etherscan_connector <wallet_address>")
        sys.exit(1)

    addr = sys.argv[1]
    raw = fetch_onchain_history(addr)
    print(json.dumps(raw, indent=2))
