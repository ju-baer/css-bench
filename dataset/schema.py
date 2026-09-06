"""
css_bench.dataset.schema
==========================
Game topologies (paper §0.1) and the Pydantic schema for a single dataset
instance. See README.md "Formal Framework" for the full derivation of
Definition 0.1 (task) and Proposition 0.2 (Prisoner's Dilemma invariant).
"""
from typing import Dict

from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Game topologies: the ordinal constraint on (R, S, T, P) that defines each
# classic game. Slot A pays (R if opponent plays A, S if opponent plays B);
# Slot B pays (T if opponent plays A, P if opponent plays B).
# ---------------------------------------------------------------------------
TOPOLOGIES = {
    "prisoners_dilemma": dict(
        check=lambda R, S, T, P: T > R > P > S and 2 * R > T + S,
        description="Mutual cooperation beats mutual defection, but defection dominates individually.",
    ),
    "chicken": dict(
        check=lambda R, S, T, P: T > R > S > P,
        description="Mutual yielding is safe; unilateral defection is best; mutual defection is worst.",
    ),
    "stag_hunt": dict(
        check=lambda R, S, T, P: R > T > P > S,
        description="Cooperation is payoff-dominant but risk-dominated by defection.",
    ),
}


class GameInstanceSchema(BaseModel):
    """One rendered game instance: a specific payoff matrix, topology,
    semantic category, and projection (canon/counter). `correct_slot` is
    ground truth under Definition 0.1 (average-payoff criterion) and is
    entirely a property of `payoffs` -- never of the labels."""

    instance_id: str
    topology_type: str
    payoffs: Dict[str, int]
    label_A: str
    label_B: str
    correct_slot: str          # "A" or "B" -- ground truth, from payoffs only
    projection: str            # "canon" or "counter"
    semantic_category: str     # key into css_bench.dataset.taxonomy.SEMANTIC_TAXONOMY
    payoff_margin: int         # |avg_A - avg_B|, the numerical-difficulty factor
    seed: int

    @field_validator("projection")
    @classmethod
    def _check_proj(cls, v):
        assert v in ("canon", "counter")
        return v

    @property
    def correct_label(self) -> str:
        """The label sitting on the numerically-optimal slot."""
        return self.label_A if self.correct_slot == "A" else self.label_B

    @property
    def incorrect_label(self) -> str:
        """The label sitting on the numerically-suboptimal slot."""
        return self.label_B if self.correct_slot == "A" else self.label_A
