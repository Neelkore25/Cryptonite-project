"""
cryptonite.engine.alert_engine
----------------------------------
Combines the two soft-computing/AIML signals into one final verdict:

  1. Fuzzy Trust Score   (0-100, interpretable, rule-based)   -- WHY it's risky
  2. ANN Bot Probability (0-1, statistical pattern match)       -- HOW LIKELY it's automated

final_score = 0.5 * fuzzy_trust_score + 0.5 * (100 * (1 - ann_bot_probability))

This hybrid (neuro-fuzzy) combination means a single confused signal
doesn't unfairly tank an account's standing, while both models agreeing
produces a confident, explainable verdict.
"""

from dataclasses import dataclass, field
from typing import Dict

from cryptonite.engine.feature_engineering import to_fuzzy_inputs, to_ml_vector
from cryptonite.engine.fuzzy_trust import FuzzyTrustEngine
from cryptonite.engine.ml_bot_detector import BotDetector

RISK_LEVELS = [
    (75, "TRUSTED",        "✅", "Account shows strong, consistent human-like activity."),
    (55, "LOW_RISK",       "🟢", "Account looks mostly legitimate with minor irregularities."),
    (35, "MEDIUM_RISK",    "🟡", "Account shows mixed signals — recommend monitoring."),
    (18, "HIGH_RISK",      "🟠", "Account shows strong bot/scam-pattern indicators."),
    (0,  "SUSPICIOUS_BOT", "🔴", "Account matches automated/scam-bot behavior closely. Alert triggered."),
]


@dataclass
class AccountReport:
    user_id: str
    final_score: float
    fuzzy_trust_score: float
    ann_bot_probability: float
    fuzzy_inputs: Dict[str, float]
    risk_level: str
    icon: str
    message: str
    reasons: list = field(default_factory=list)

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "final_score": round(self.final_score, 1),
            "fuzzy_trust_score": round(self.fuzzy_trust_score, 1),
            "ann_bot_probability": round(self.ann_bot_probability, 3),
            "fuzzy_inputs": {k: round(v, 1) for k, v in self.fuzzy_inputs.items()},
            "risk_level": self.risk_level,
            "icon": self.icon,
            "message": self.message,
            "reasons": self.reasons,
        }


class CryptoniteAlertEngine:
    def __init__(self, fuzzy_params=None, fuzzy_weight=0.5):
        self.fuzzy_engine = FuzzyTrustEngine(fuzzy_params)
        self.bot_detector = BotDetector()
        if BotDetector.exists():
            self.bot_detector.load()
        else:
            self.bot_detector.train(verbose=False)
            self.bot_detector.save()
        self.fuzzy_weight = fuzzy_weight

    def _risk_bucket(self, score):
        for threshold, level, icon, msg in RISK_LEVELS:
            if score >= threshold:
                return level, icon, msg
        return RISK_LEVELS[-1][1:]

    def _explain(self, fuzzy_inputs, ann_prob):
        reasons = []
        if fuzzy_inputs["account_age"] < 30:
            reasons.append("Account is very new.")
        if fuzzy_inputs["txn_regularity"] < 30:
            reasons.append("Transaction timing/value pattern looks robotic and unnaturally regular.")
        if fuzzy_inputs["network_diversity"] < 30:
            reasons.append("Interacts with a very narrow, repetitive set of wallets and mostly at odd hours.")
        if fuzzy_inputs["verification"] < 30:
            reasons.append("No KYC verification and/or suspicious follower-to-following ratio and reports on file.")
        if ann_prob >= 0.5:
            reasons.append("Neural network classifies this activity pattern as BOT-like.")
        if not reasons:
            reasons.append("No significant risk indicators detected.")
        return reasons

    def analyze(self, user_id: str, raw_features: dict) -> AccountReport:
        fuzzy_inputs = to_fuzzy_inputs(raw_features)
        trust_score = self.fuzzy_engine.score(**fuzzy_inputs)

        ml_vector = to_ml_vector(raw_features)
        bot_prob = self.bot_detector.predict_proba(ml_vector)

        final_score = (self.fuzzy_weight * trust_score +
                        (1 - self.fuzzy_weight) * (100 * (1 - bot_prob)))

        level, icon, msg = self._risk_bucket(final_score)
        reasons = self._explain(fuzzy_inputs, bot_prob)

        return AccountReport(
            user_id=user_id,
            final_score=final_score,
            fuzzy_trust_score=trust_score,
            ann_bot_probability=bot_prob,
            fuzzy_inputs=fuzzy_inputs,
            risk_level=level,
            icon=icon,
            message=msg,
            reasons=reasons,
        )


if __name__ == "__main__":
    from cryptonite.data.simulator import fetch_account_history

    engine = CryptoniteAlertEngine()
    for uid in ["alice_trader_2019", "quickflip_bot_007", "0xAnon_new_wallet", "grandma_hodler"]:
        raw = fetch_account_history(uid)
        report = engine.analyze(uid, raw)
        print(f"\n{report.icon} {uid}: {report.risk_level}  (final={report.final_score:.1f}, "
              f"fuzzy={report.fuzzy_trust_score:.1f}, ann_bot_prob={report.ann_bot_probability:.2f})")
        for r in report.reasons:
            print("   -", r)
