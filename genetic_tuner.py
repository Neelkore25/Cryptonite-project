"""
cryptonite.engine.genetic_tuner
----------------------------------
SOFT COMPUTING CONCEPT #2: GENETIC ALGORITHM (evolutionary optimization)

Hand-picked fuzzy membership-function breakpoints (in fuzzy_trust.py's
default_params) are a reasonable starting guess, but not necessarily
optimal. A Genetic Algorithm evolves a population of candidate breakpoint
sets, using the ML-labelled dataset as ground truth, to find parameters
that make the Fuzzy Trust Engine's output best separate bots from humans
(low trust score <-> bot, high trust score <-> human).

This is the classic "Neuro-Fuzzy-Genetic" hybrid soft-computing pattern:
  Fuzzy Logic   -> interpretable trust reasoning
  Genetic Algo  -> tunes the fuzzy system's parameters
  Neural Net    -> independent statistical classifier
  (combined by engine/alert_engine.py into one hybrid decision)

Chromosome encoding (12 genes, each a breakpoint on the 0-100 universe):
  [age_lo, age_mid, age_hi,
   reg_lo, reg_mid, reg_hi,
   div_lo, div_mid, div_hi,
   ver_lo, ver_mid, ver_hi]
(the trapezoid consequent sets for trust_low/medium/high are kept fixed --
 only the antecedent membership functions are evolved.)
"""

import numpy as np

from cryptonite.data.simulator import generate_dataset
from cryptonite.engine.feature_engineering import to_fuzzy_inputs
from cryptonite.engine.fuzzy_trust import FuzzyTrustEngine

GENE_KEYS = ["account_age", "txn_regularity", "network_diversity", "verification"]


def _chromosome_to_params(chromo):
    params = FuzzyTrustEngine.default_params()
    for i, key in enumerate(GENE_KEYS):
        lo, mid, hi = sorted(chromo[i * 3:i * 3 + 3])
        lo = float(np.clip(lo, 1, 40))
        mid = float(np.clip(mid, lo + 5, 90))
        hi = float(np.clip(hi, mid + 5, 99))
        params[key] = (lo, mid, hi)
    return params


def _random_chromosome(rng):
    chromo = []
    for _ in GENE_KEYS:
        lo = rng.uniform(10, 35)
        mid = rng.uniform(lo + 10, 70)
        hi = rng.uniform(mid + 10, 95)
        chromo += [lo, mid, hi]
    return np.array(chromo)


def _fitness(chromo, sample_df, cache):
    """
    Fitness = how well the resulting trust_score separates humans (label 0,
    should score HIGH) from bots (label 1, should score LOW), measured as
    the AUC-like rank separation (Mann-Whitney U statistic), computed
    against a fixed pre-fuzzified input sample for speed.
    """
    key = tuple(np.round(chromo, 2))
    if key in cache:
        return cache[key]

    params = _chromosome_to_params(chromo)
    engine = FuzzyTrustEngine(params)

    scores = []
    for row in sample_df.itertuples():
        s = engine.score(row.account_age, row.txn_regularity,
                          row.network_diversity, row.verification)
        scores.append(s)
    scores = np.array(scores)
    labels = sample_df["is_bot"].values

    human_scores = scores[labels == 0]
    bot_scores = scores[labels == 1]
    if len(human_scores) == 0 or len(bot_scores) == 0:
        return 0.0

    # Rank-based separation (equivalent to AUC): fraction of (human, bot)
    # pairs where human_score > bot_score.
    diffs = human_scores[:, None] - bot_scores[None, :]
    auc = float(np.mean(diffs > 0) + 0.5 * np.mean(diffs == 0))
    cache[key] = auc
    return auc


def run_genetic_algorithm(pop_size=24, generations=20, sample_size=300,
                           mutation_rate=0.25, mutation_scale=6.0, seed=7,
                           verbose=True):
    rng = np.random.default_rng(seed)

    raw = generate_dataset(n_humans=sample_size // 2, n_bots=sample_size // 2, seed=seed)
    fuzzy_rows = raw.apply(lambda r: to_fuzzy_inputs(r.to_dict()), axis=1, result_type="expand")
    sample_df = fuzzy_rows.copy()
    sample_df["is_bot"] = raw["is_bot"].values

    population = [_random_chromosome(rng) for _ in range(pop_size)]
    cache = {}
    history = []

    for gen in range(generations):
        fitnesses = np.array([_fitness(c, sample_df, cache) for c in population])
        order = np.argsort(-fitnesses)
        population = [population[i] for i in order]
        fitnesses = fitnesses[order]
        history.append(float(fitnesses[0]))

        if verbose:
            print(f"Gen {gen+1:2d}/{generations}  best_fitness={fitnesses[0]:.4f}  "
                  f"mean_fitness={fitnesses.mean():.4f}")

        # --- Selection: elitism (top 20%) survives unchanged ---
        n_elite = max(2, pop_size // 5)
        next_gen = population[:n_elite]

        # --- Crossover: roulette-wheel selection of parents from top half ---
        parent_pool = population[:pop_size // 2]
        parent_fit = fitnesses[:pop_size // 2]
        probs = (parent_fit - parent_fit.min() + 1e-6)
        probs = probs / probs.sum()

        while len(next_gen) < pop_size:
            p1, p2 = rng.choice(len(parent_pool), size=2, replace=True, p=probs)
            cut = rng.integers(1, len(GENE_KEYS) * 3 - 1)
            child = np.concatenate([parent_pool[p1][:cut], parent_pool[p2][cut:]])

            # --- Mutation ---
            if rng.random() < mutation_rate:
                gene_idx = rng.integers(0, len(child))
                child[gene_idx] += rng.normal(0, mutation_scale)
                child = np.clip(child, 1, 99)

            next_gen.append(child)

        population = next_gen

    best_chromo = population[0]
    best_fitness = _fitness(best_chromo, sample_df, cache)
    best_params = _chromosome_to_params(best_chromo)

    if verbose:
        print("\nBest evolved membership-function breakpoints:")
        for k in GENE_KEYS:
            print(f"  {k:20s}: {tuple(round(v, 1) for v in best_params[k])}")
        print(f"Final separation fitness (rank-AUC): {best_fitness:.4f}")

    return best_params, history


if __name__ == "__main__":
    run_genetic_algorithm()
