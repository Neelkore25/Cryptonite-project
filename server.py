"""
cryptonite.server
--------------------
Small local backend so the dashboard can look up REAL wallets with just a
click -- no console commands per check. The Etherscan API key stays here,
loaded from a local .env file, and is never sent to or exposed in the
browser.

One-time setup:
    cp .env.example .env          # then paste your key into .env
    pip install -r requirements.txt

Run it:
    python server.py

Then open http://localhost:5000 in your browser. That's the only command
you'll ever need to type -- everything else (including live wallet scans)
happens by clicking in the page.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from cryptonite.data.etherscan_connector import fetch_onchain_history, EtherscanError
from cryptonite.engine.alert_engine import CryptoniteAlertEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)  # allow the dashboard page to call this API even if opened from a different origin

_engine = None


def get_engine():
    """Lazy-load the alert engine (and its trained models) once, on first request."""
    global _engine
    if _engine is None:
        fuzzy_params = None
        ga_path = os.path.join(BASE_DIR, "models", "ga_fuzzy_params.json")
        if os.path.exists(ga_path):
            with open(ga_path) as f:
                fuzzy_params = {k: tuple(v) for k, v in json.load(f).items()}
        _engine = CryptoniteAlertEngine(fuzzy_params=fuzzy_params)
    return _engine


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.route("/api/health")
def health():
    has_key = bool(os.environ.get("ETHERSCAN_API_KEY"))
    return jsonify({"status": "ok", "etherscan_key_configured": has_key})


@app.route("/api/check-live")
def check_live():
    """
    GET /api/check-live?address=0x...
    Fetches REAL on-chain history for the address and runs it through the
    fuzzy + GA + ANN pipeline. The API key never leaves this server.
    """
    address = request.args.get("address", "").strip()
    if not address:
        return jsonify({"error": "Missing ?address= parameter"}), 400

    try:
        raw = fetch_onchain_history(address)
    except EtherscanError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Unexpected error fetching on-chain data: {e}"}), 500

    engine = get_engine()
    report = engine.analyze(address, raw)
    result = report.to_dict()
    result["raw_features"] = {
        k: (round(v, 4) if isinstance(v, float) else v) for k, v in raw.items()
    }
    return jsonify(result)


if __name__ == "__main__":
    from cryptonite.data.etherscan_connector import _ENV_PATH

    key = os.environ.get("ETHERSCAN_API_KEY")
    print("=" * 60)
    if _ENV_PATH:
        print(f".env path checked : {_ENV_PATH}")
        print(f".env file exists  : {_ENV_PATH.exists()}")
    if key:
        masked = key[:4] + "…" + key[-4:] if len(key) > 8 else "****"
        print(f"Etherscan API key : found ({masked})")
    else:
        print("Etherscan API key : NOT FOUND")
        print("  -> copy .env.example to .env in the cryptonite/ folder,")
        print("     paste your key in, save it, and re-run `python server.py`.")
    print("=" * 60)

    print("Loading models (this trains them on first run if not already saved)...")
    get_engine()
    print("\nCryptonite server ready -> open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
