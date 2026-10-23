"""
cryptonite.engine.fuzzy_trust
-------------------------------
SOFT COMPUTING CONCEPT #1: FUZZY LOGIC (Mamdani Inference System)

Human trust is not binary ("trusted" / "not trusted") and account signals
are noisy and imprecise -- exactly the kind of problem fuzzy logic is meant
for, instead of crisp if/else thresholds.

We fuzzify four crisp account-history features into linguistic terms
(low / medium / high), run them through a Mamdani rule base, and defuzzify
(centroid method) into a crisp Trust Score in [0, 100].

Inputs:
    account_age      -> normalized account age (days)
    txn_regularity    -> normalized transaction pattern naturalness
                          (derived from counterparty diversity + value variance)
    network_diversity -> normalized counterparty diversity
    verification      -> normalized KYC / social trust signal

Output:
    trust_score in [0, 100]
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyTrustEngine:
    def __init__(self, params=None):
        """
        params: optional dict overriding the default membership-function
        breakpoints. This is exactly what the Genetic Algorithm (see
        engine/genetic_tuner.py) evolves to improve accuracy -- it hands
        this class a tuned `params` dict instead of the hand-picked defaults.
        """
        self.params = params or self.default_params()
        self._build_system()

    @staticmethod
    def default_params():
        # Each entry is the (low_peak, mid_peak, high_peak) trapezoid/triangle
        # breakpoints on a 0-100 normalized universe for that variable.
        return {
            "account_age":      (20, 50, 80),
            "txn_regularity":   (25, 50, 75),
            "network_diversity":(20, 50, 80),
            "verification":     (25, 55, 80),
            "trust_low":        (0, 0, 35, 50),
            "trust_medium":     (30, 50, 50, 70),
            "trust_high":       (55, 75, 100, 100),
        }

    def _tri_terms(self, universe, breakpoints):
        lo, mid, hi = breakpoints
        low = fuzz.trimf(universe, [0, 0, mid])
        med = fuzz.trimf(universe, [lo, mid, hi])
        high = fuzz.trimf(universe, [mid, 100, 100])
        return low, med, high

    def _build_system(self):
        p = self.params
        universe = np.arange(0, 101, 1)

        self.age = ctrl.Antecedent(universe, "account_age")
        self.reg = ctrl.Antecedent(universe, "txn_regularity")
        self.div = ctrl.Antecedent(universe, "network_diversity")
        self.ver = ctrl.Antecedent(universe, "verification")
        self.trust = ctrl.Consequent(universe, "trust_score")

        for var, key in [(self.age, "account_age"), (self.reg, "txn_regularity"),
                          (self.div, "network_diversity"), (self.ver, "verification")]:
            low, med, high = self._tri_terms(universe, p[key])
            var["low"] = low
            var["medium"] = med
            var["high"] = high

        tl, tm, th = p["trust_low"], p["trust_medium"], p["trust_high"]
        self.trust["low"] = fuzz.trapmf(universe, list(tl))
        self.trust["medium"] = fuzz.trapmf(universe, list(tm))
        self.trust["high"] = fuzz.trapmf(universe, list(th))

        rules = [
            # Strong trust: everything checks out
            ctrl.Rule(self.age["high"] & self.reg["high"] & self.div["high"] & self.ver["high"], self.trust["high"]),
            ctrl.Rule(self.age["high"] & self.reg["high"] & self.ver["high"], self.trust["high"]),
            ctrl.Rule(self.div["high"] & self.ver["high"] & self.reg["high"], self.trust["high"]),
            # Strong distrust: young, robotic, unverified, low diversity
            ctrl.Rule(self.age["low"] & self.reg["low"] & self.div["low"], self.trust["low"]),
            ctrl.Rule(self.reg["low"] & self.ver["low"], self.trust["low"]),
            ctrl.Rule(self.div["low"] & self.reg["low"], self.trust["low"]),
            ctrl.Rule(self.age["low"] & self.ver["low"] & self.div["low"], self.trust["low"]),
            # Mixed / ambiguous signals -> medium, needs monitoring
            ctrl.Rule(self.age["medium"] & self.reg["medium"], self.trust["medium"]),
            ctrl.Rule(self.age["high"] & self.reg["low"], self.trust["medium"]),
            ctrl.Rule(self.ver["medium"] & self.div["medium"], self.trust["medium"]),
            ctrl.Rule(self.age["low"] & self.reg["high"] & self.ver["high"], self.trust["medium"]),
            ctrl.Rule(self.div["high"] & self.reg["low"], self.trust["medium"]),
        ]

        self.system = ctrl.ControlSystem(rules)

    def score(self, account_age, txn_regularity, network_diversity, verification):
        """All four inputs must already be normalized to 0-100."""
        sim = ctrl.ControlSystemSimulation(self.system)
        sim.input["account_age"] = float(np.clip(account_age, 0, 100))
        sim.input["txn_regularity"] = float(np.clip(txn_regularity, 0, 100))
        sim.input["network_diversity"] = float(np.clip(network_diversity, 0, 100))
        sim.input["verification"] = float(np.clip(verification, 0, 100))
        try:
            sim.compute()
            return float(sim.output["trust_score"])
        except Exception:
            # No rule fired strongly enough (edge case) -> fall back to a
            # weighted-average heuristic so the pipeline never crashes.
            return float(np.mean([account_age, txn_regularity, network_diversity, verification]))


if __name__ == "__main__":
    engine = FuzzyTrustEngine()
    print("Human-like:", engine.score(80, 85, 70, 90))
    print("Bot-like:  ", engine.score(10, 8, 5, 5))
    print("Ambiguous: ", engine.score(45, 50, 55, 40))
