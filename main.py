"""
Cryptonite -- Crypto Scam/Bot Account Trust & Alert System
=============================================================
Hybrid Soft Computing + AIML pipeline:
    Fuzzy Logic (Mamdani)  ->  interpretable trust scoring
    Genetic Algorithm       ->  tunes the fuzzy membership functions
    Artificial Neural Net   ->  statistical bot/human classifier
    Hybrid Alert Engine     ->  fuses both into one final verdict + alert

Usage:
    python3 main.py --train                 # retrain the ANN + run the GA
    python3 main.py --check <user_id>        # analyze one account
    python3 main.py --demo                   # analyze a bundled demo batch
    python3 main.py --batch-report out.json  # analyze demo batch -> JSON
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from cryptonite.data.simulator import fetch_account_history
from cryptonite.engine.alert_engine import CryptoniteAlertEngine
from cryptonite.engine.ml_bot_detector import BotDetector
from cryptonite.engine.genetic_tuner import run_genetic_algorithm

DEMO_ACCOUNTS = [
    "alice_trader_2019",
    "quickflip_bot_007",
    "0xAnon_new_wallet",
    "grandma_hodler",
    "airdrop_hunter_x1000",
    "satoshi_disciple",
    "pump_signal_caller",
    "long_term_hodler_99",
]


def cmd_train():
    print("Step 1/2: Training ANN bot detector on synthetic labelled dataset...")
    det = BotDetector()
    det.train()
    det.save()

    print("\nStep 2/2: Running Genetic Algorithm to tune fuzzy membership functions...")
    best_params, history = run_genetic_algorithm()

    out_path = os.path.join(os.path.dirname(__file__), "models", "ga_fuzzy_params.json")
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved GA-tuned fuzzy parameters to {out_path}")


def _load_engine():
    ga_path = os.path.join(os.path.dirname(__file__), "models", "ga_fuzzy_params.json")
    fuzzy_params = None
    if os.path.exists(ga_path):
        with open(ga_path) as f:
            fuzzy_params = json.load(f)
            fuzzy_params = {k: tuple(v) for k, v in fuzzy_params.items()}
    return CryptoniteAlertEngine(fuzzy_params=fuzzy_params)


def cmd_check(user_id):
    engine = _load_engine()
    raw = fetch_account_history(user_id)
    report = engine.analyze(user_id, raw)
    _print_report(report)


def cmd_check_live(address, api_key=None):
    """Analyze a REAL Ethereum wallet using live on-chain data (Etherscan)."""
    from cryptonite.data.etherscan_connector import fetch_onchain_history, EtherscanError

    print(f"Fetching live on-chain history for {address} from Etherscan...")
    try:
        raw = fetch_onchain_history(address, api_key=api_key)
    except EtherscanError as e:
        print(f"Error: {e}")
        return

    print(f"Fetched {raw['txn_count']} real transactions "
          f"spanning {raw['account_age_days']:.1f} days.\n")

    engine = _load_engine()
    report = engine.analyze(address, raw)
    _print_report(report)


def cmd_demo():
    engine = _load_engine()
    for uid in DEMO_ACCOUNTS:
        raw = fetch_account_history(uid)
        report = engine.analyze(uid, raw)
        _print_report(report)


def cmd_batch_report(path):
    engine = _load_engine()
    results = []
    for uid in DEMO_ACCOUNTS:
        raw = fetch_account_history(uid)
        report = engine.analyze(uid, raw)
        d = report.to_dict()
        d["raw_features"] = {k: (round(float(v), 3) if isinstance(v, float) else int(v)) for k, v in raw.items()}
        results.append(d)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote batch report for {len(results)} accounts to {path}")


def _print_report(report):
    print(f"\n{report.icon}  {report.user_id}")
    print(f"   Risk level        : {report.risk_level}")
    print(f"   Final trust score : {report.final_score:.1f} / 100")
    print(f"   Fuzzy trust score : {report.fuzzy_trust_score:.1f} / 100  (soft-computing / fuzzy logic)")
    print(f"   ANN prediction     : {'BOT' if report.ann_bot_probability >= 0.5 else 'HUMAN'}  (neural network)")
    print(f"   {report.message}")
    for r in report.reasons:
        print(f"     - {r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cryptonite crypto scam/bot account alert system")
    parser.add_argument("--train", action="store_true", help="train ANN + run GA fuzzy tuning")
    parser.add_argument("--check", metavar="USER_ID", help="analyze a single (simulated) account")
    parser.add_argument("--check-live", metavar="ETH_ADDRESS",
                         help="analyze a REAL Ethereum wallet via the Etherscan API "
                              "(needs ETHERSCAN_API_KEY env var or --api-key)")
    parser.add_argument("--api-key", metavar="KEY", help="Etherscan API key (overrides ETHERSCAN_API_KEY env var)")
    parser.add_argument("--demo", action="store_true", help="run the bundled demo batch")
    parser.add_argument("--batch-report", metavar="PATH", help="write demo batch results as JSON")
    args = parser.parse_args()

    if args.train:
        cmd_train()
    elif args.check_live:
        cmd_check_live(args.check_live, api_key=args.api_key)
    elif args.check:
        cmd_check(args.check)
    elif args.demo:
        cmd_demo()
    elif args.batch_report:
        cmd_batch_report(args.batch_report)
    else:
        parser.print_help()
