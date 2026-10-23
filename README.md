# Cryptonite
### Crypto scam / bot account trust & alert system — a hybrid Soft Computing + AIML project

Cryptonite reads a crypto account's activity history and produces an
explainable **Trust Score** and **risk alert**, by combining two different
computational paradigms rather than relying on brittle hand-written
threshold rules:

| Layer | Concept | Role |
|---|---|---|
| Fuzzy Logic | **Soft Computing** | Interpretable reasoning over imprecise, human-scale signals ("is this account *fairly* new?") |
| Genetic Algorithm | **Soft Computing** | Evolves/optimizes the fuzzy system's membership-function parameters |
| Artificial Neural Network | **AIML** | Statistical pattern classifier trained on labelled bot/human account histories |
| Hybrid Alert Engine | **Neuro-Fuzzy-Genetic fusion** | Combines both signals into one final, explained verdict |

This "Neuro-Fuzzy-Genetic" hybrid pattern is one of the classic architectures
taught in Soft Computing courses: fuzzy logic gives interpretability, the
genetic algorithm tunes it automatically instead of by hand, and the neural
network adds a statistically learned second opinion that fuzzy rules alone
can't capture.

---

## 1. Why fuzzy logic?

Whether an account is "old enough" or "verified enough" to be trusted isn't
a crisp yes/no — it's a matter of degree. `engine/fuzzy_trust.py` implements
a **Mamdani fuzzy inference system** with four linguistic input variables:

- `account_age` — how long the account has existed
- `txn_regularity` — how naturally irregular the transaction timing/values are (bots are suspiciously regular)
- `network_diversity` — how many distinct counterparties the account interacts with, and whether activity clusters at odd hours
- `verification` — KYC status, healthy follower/following ratio, absence of community reports

Each is fuzzified into **low / medium / high** membership via triangular
membership functions, run through a rule base (e.g. *"IF age is high AND
regularity is high AND verification is high THEN trust is high"*), and
defuzzified via the **centroid method** into a crisp Trust Score in `[0,100]`.

## 2. Why a genetic algorithm?

The fuzzy system's membership-function breakpoints (where "low" ends and
"medium" begins, etc.) were initially hand-picked guesses. `engine/genetic_tuner.py`
evolves them automatically:

- **Chromosome**: 12 real-valued genes = 3 breakpoints × 4 input variables
- **Fitness**: rank-based separation (equivalent to AUC) between the fuzzy
  trust scores assigned to known humans vs. known bots — higher fitness
  means the fuzzy system separates the two classes better
- **Selection**: elitism (top 20% survive) + roulette-wheel selection of parents
- **Crossover**: single-point crossover between two parent chromosomes
- **Mutation**: Gaussian perturbation of a random gene, at a configurable rate

Running it (`python main.py --train`) converges to ~0.9997 separation
fitness within 20 generations and saves the tuned parameters to
`models/ga_fuzzy_params.json`, which the alert engine loads automatically.

## 3. Why a neural network?

Fuzzy rules encode *domain intuition*. The **Multi-Layer Perceptron**
(`engine/ml_bot_detector.py`, architecture `15 → 32 → 16 → 1`, ReLU hidden
layers, trained with Adam) instead *learns* statistical patterns directly
from 1,500 synthetic human + 1,500 synthetic bot account histories (with
8% overlap/noise so the task isn't trivially separable). A Random Forest is
trained alongside purely as a classical-ML comparison baseline — the ANN is
the model actually used at inference time.

Typical held-out test performance: **~98% accuracy, ROC-AUC ≈ 0.999**.

## 4. The hybrid alert engine

```
final_score = 0.5 × fuzzy_trust_score + 0.5 × 100×(1 − ann_bot_probability)
```

The final score is bucketed into five risk tiers (`TRUSTED`, `LOW_RISK`,
`MEDIUM_RISK`, `HIGH_RISK`, `SUSPICIOUS_BOT`), each with a plain-English
explanation and a bullet list of the specific signals that drove the
verdict — so the alert is never a black box.

---

## Project structure

```
cryptonite/
├── data/
│   ├── simulator.py           # synthetic account-history generator (used for training/demo)
│   └── etherscan_connector.py # REAL on-chain data connector (Etherscan API)
├── engine/
│   ├── feature_engineering.py  # raw features -> fuzzy inputs + ML vector
│   ├── fuzzy_trust.py          # Mamdani fuzzy inference system
│   ├── genetic_tuner.py        # GA that tunes the fuzzy system
│   ├── ml_bot_detector.py      # ANN (+ RF baseline) bot classifier
│   └── alert_engine.py         # fuses fuzzy + ANN into final verdict
├── models/                   # trained ANN weights, scaler, GA-tuned params (generated)
├── reports/                  # generated JSON batch reports
├── main.py                   # CLI entry point
├── server.py                  # local Flask backend: serves the dashboard + /api/check-live
├── dashboard.html             # frontend (works standalone for demo IDs; needs server.py for live wallets)
├── .env.example                # copy to .env and add your Etherscan key
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

# 1. One-time setup: put your free Etherscan key in a local .env file
cp .env.example .env
# then open .env and paste your key in (get one free at https://etherscan.io/apis)

# 2. Train the ANN and run the GA to tune the fuzzy engine (creates models/)
python main.py --train

# 3. Start the local web app (serves the dashboard + the live-lookup API)
python server.py
# -> open http://localhost:5000 in your browser
```

That's the only console command you'll need after setup — from there, everything
happens by clicking in the browser:
- Type a demo ID (`quickflip_bot_007`, `alice_trader_2019`, …) → analyzed
  instantly, entirely in the browser, no server round-trip.
- Paste a **real** `0x…` wallet address → the page calls the local server's
  `/api/check-live` endpoint, which fetches that wallet's actual transaction
  history from Etherscan (using the key from your `.env`) and runs it through
  the same fuzzy + GA + ANN pipeline.

The API key never leaves `server.py` — it's read from `.env` on the backend
and is never sent to or visible in the browser.

### Command-line alternatives (optional)
Everything above is also available without the web app, if you prefer:
```bash
python main.py --check some_wallet_id            # simulated account
python main.py --check-live 0xd8dA6BF269...       # real wallet, via CLI
python main.py --demo                             # bundled demo batch
python main.py --batch-report reports/out.json    # JSON batch export
```

`dashboard.html` can also be opened directly as a static file (no server) —
in that mode, demo-ID lookups still work client-side, but real `0x…`
addresses will show a "backend not reachable" message until you run
`python server.py`.

## Plugging in a real exchange / wallet API

Everything downstream of the raw feature dict is agnostic to where it came
from — the fuzzy engine, ANN, and alert engine never change. This project
ships **two** data sources behind that same interface:

### Simulated data (default)
`data/simulator.py:fetch_account_history()` — synthetic accounts, used for
training the ANN and for the `--demo` / `--check` commands.

### Real on-chain data (Etherscan)
`data/etherscan_connector.py:fetch_onchain_history()` — pulls an actual
Ethereum wallet's real transaction history from the free [Etherscan
API](https://etherscan.io/apis) and computes the exact same feature dict
from real timestamps, values, and counterparties. No OAuth, no user login —
on-chain data is public, so you can score any wallet address immediately.

```bash
pip install -r requirements.txt        # now includes `requests`
export ETHERSCAN_API_KEY=your_free_key_here

python main.py --check-live 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

This fetches the wallet's real transaction list, derives account age,
transaction regularity, counterparty diversity, night-activity ratio, and
deposit/withdrawal balance from it, and runs the exact same fuzzy + GA +
ANN pipeline used everywhere else in the project.

**Honest limitation:** on-chain data alone can't see KYC status, social
graph, or community reports — those fields are set to neutral defaults
(`data/etherscan_connector.py:UNAVAILABLE_DEFAULTS`) rather than zero, so a
clean wallet isn't unfairly penalized just for lacking off-chain signals.
For a capstone demo this is a reasonable, clearly-documented simplification;
a production system would need those signals from an exchange's own API
under the account holder's consent, which is a materially different (and
much more involved) integration than the public on-chain case above.

### Wiring in your own data source
Write a function that returns a dict with the keys listed in
`FEATURE_COLUMNS` (`data/simulator.py`), and pass it straight to
`CryptoniteAlertEngine.analyze(user_id, raw_features)` — that's the entire
integration surface.

## Limitations / notes for submission

- All training data is **synthetically simulated** (`data/simulator.py`),
  not real exchange data — clearly documented as a stand-in for a real API.
- The 98%+ ANN accuracy reflects the synthetic dataset's separability; real
  deployments would need labelled real-world bot/scam reports.
- This is an educational/demo system, not a production fraud pipeline —
  it should be one signal among several in a real trust & safety stack.
