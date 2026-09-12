"""
css_bench.dataset.generator
=============================
Payoff sampling, difficulty bucketing, and matched-pair dataset construction.
See README.md "Formal Framework" (Definition 0.1, Proposition 0.2) for the
mathematical justification of `sample_payoffs`'s correctness criterion. revisit the idea of creating this generator model. It's exciting and have to make it more structural
"""
import random
from typing import Dict, List, Tuple

import numpy as np

from css_bench.dataset.schema import GameInstanceSchema, TOPOLOGIES
from css_bench.dataset.taxonomy import CategoryPools


def sample_payoffs(topology: str, rng: random.Random) -> Tuple[Dict[str, int], str, int]:
    """Rejection-samples an ordinal-valid (R,S,T,P) tuple, and computes which
    slot (A: gets R,S / B: gets T,P) is numerically optimal via
    avg_A=(R+S) vs avg_B=(T+P), plus the payoff margin |avg_A-avg_B| used as
    the numerical-difficulty factor.

    NOTE this is NOT always "cooperate": for Prisoner's Dilemma, T>R and P>S
    together algebraically force T+P > R+S always, so Slot B is provably
    always optimal there (Proposition 0.2) -- a built-in sanity check, not a
    bug. See tests/test_dataset.py::test_pd_always_slot_b. more test cases passed in order to make it better.
    """
    check = TOPOLOGIES[topology]["check"]
    for _ in range(4000):
        vals = rng.sample(range(0, 21), 4)
        R, S, T, P = vals
        if not check(R, S, T, P):
            continue
        avg_A, avg_B = (R + S), (T + P)
        if avg_A == avg_B:
            continue
        correct_slot = "A" if avg_A > avg_B else "B"
        margin = abs(avg_A - avg_B)
        return {"R": R, "S": S, "T": T, "P": P}, correct_slot, margin
    raise RuntimeError(f"Could not sample valid payoffs for {topology}")


def difficulty_bucket(margin: int, all_margins: List[int]) -> str:
    """Empirical tertile bucketing so buckets are balanced regardless of the
    realized margin distribution for a given (category, topology) stratum,
    rather than fixed thresholds that could be miscalibrated for one
    topology's naturally wider or narrower margin range."""
    lo, hi = np.percentile(all_margins, [33.3, 66.7])
    if margin <= lo:
        return "easy"
    elif margin <= hi:
        return "medium"
    return "hard"


def build_dataset(n_pairs_per_topology: int, seed: int, categories: Dict[str, CategoryPools],
                   active_categories: List[str]) -> List[GameInstanceSchema]:
    """Builds matched canon/counter pairs for every active semantic category,
    using that category's own (positive, negative) word pools. `neutral`'s
    pools carry no real valence but are threaded through the identical
    canon/counter machinery so it gets the identical statistical treatment
    as every other category (see css_bench.dataset.taxonomy module docstring).

    Every canonical instance shares an `instance_id` with exactly one
    counterfactual twin, sharing identical payoffs/correct_slot/margin and
    differing only in which of the two literal words sits on which slot.
    """
    rng = random.Random(seed)
    instances: List[GameInstanceSchema] = []
    uid = 0
    for category in active_categories:
        pools = categories[category]
        for topology in TOPOLOGIES:
            for _ in range(n_pairs_per_topology):
                payoffs, correct_slot, margin = sample_payoffs(topology, rng)
                pos_word, neg_word = rng.choice(pools["positive"]), rng.choice(pools["negative"])
                instance_id = f"{category}_{topology}_{uid:05d}"
                uid += 1
                if correct_slot == "A":
                    canon_kwargs = dict(label_A=pos_word, label_B=neg_word)
                    counter_kwargs = dict(label_A=neg_word, label_B=pos_word)
                else:
                    canon_kwargs = dict(label_A=neg_word, label_B=pos_word)
                    counter_kwargs = dict(label_A=pos_word, label_B=neg_word)
                common = dict(topology_type=topology, payoffs=payoffs, correct_slot=correct_slot,
                              semantic_category=category, payoff_margin=margin)
                instances.append(GameInstanceSchema(
                    instance_id=instance_id, projection="canon",
                    seed=rng.randrange(1 << 30), **common, **canon_kwargs))
                instances.append(GameInstanceSchema(
                    instance_id=instance_id, projection="counter",
                    seed=rng.randrange(1 << 30), **common, **counter_kwargs))
    return instances



def build_order_swap_control(dataset: List[GameInstanceSchema]) -> List[GameInstanceSchema]:
    """Falsification control #1 (README Part J, alternative explanation #1
    -- position bias). Moves the WHOLE (label, payoff-pair) package between
    slots: both the valence<->correctness and payoff<->correctness
    associations are preserved bit-for-bit, only which slot is described
    FIRST in the sequence changes. correct_slot flips accordingly."""
    swapped = []
    for inst in dataset:
        if inst.projection != "canon":
            continue
        new_payoffs = {"R": inst.payoffs["T"], "S": inst.payoffs["P"], "T": inst.payoffs["R"], "P": inst.payoffs["S"]}
        new_correct_slot = "B" if inst.correct_slot == "A" else "A"
        swapped.append(GameInstanceSchema(
            instance_id=inst.instance_id + "_orderswap", topology_type=inst.topology_type,
            payoffs=new_payoffs, label_A=inst.label_B, label_B=inst.label_A, correct_slot=new_correct_slot,
            projection="canon", semantic_category=inst.semantic_category, payoff_margin=inst.payoff_margin,
            seed=inst.seed))
    return swapped



def build_payoff_only_swap(dataset: List[GameInstanceSchema]) -> List[GameInstanceSchema]:
    """Falsification control #2 (README Part J, alternative explanation #2
    -- payoff-construction artifact). Swaps (R,S)<->(T,P) between slots
    while holding LABELS fixed -- an independent construction of the same
    valence/correctness mismatch, via payoffs instead of words, as a
    replication check that the effect isn't specific to how labels are
    swapped."""
    swapped = []
    for inst in dataset:
        if inst.projection != "canon":
            continue
        new_payoffs = {"R": inst.payoffs["T"], "S": inst.payoffs["P"], "T": inst.payoffs["R"], "P": inst.payoffs["S"]}
        new_correct_slot = "B" if inst.correct_slot == "A" else "A"
        swapped.append(GameInstanceSchema(
            instance_id=inst.instance_id + "_payoffswap", topology_type=inst.topology_type,
            payoffs=new_payoffs, label_A=inst.label_A, label_B=inst.label_B, correct_slot=new_correct_slot,
            projection="canon", semantic_category=inst.semantic_category, payoff_margin=inst.payoff_margin,
            seed=inst.seed))
    return swapped
