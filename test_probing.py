"""
Validates the three-probe label derivation against a hand-worked truth
table -- the riskiest new logic in the three-probe redesign, since a sign
error here would silently swap what "Probe A" and "Probe B" are actually
measuring.
"""
import numpy as np

from css_bench.analysis.phase3_attention_probing import get_probe_labels


class _FakeInst:
    def __init__(self, correct_slot, projection):
        self.correct_slot = correct_slot
        self.projection = projection


def test_payoff_optimal_and_label_valence_truth_table():
    # correct_slot=A, canon -> positive word at slot A (idx 0) -> valence label 0
    inst1 = _FakeInst("A", "canon")
    # correct_slot=A, counter -> positive word at slot B (idx 1) -> valence label 1
    inst2 = _FakeInst("A", "counter")
    # correct_slot=B, canon -> positive word at slot B (idx 1) -> valence label 1
    inst3 = _FakeInst("B", "canon")
    # correct_slot=B, counter -> positive word at slot A (idx 0) -> valence label 0
    inst4 = _FakeInst("B", "counter")

    instances = [inst1, inst2, inst3, inst4]
    payoff_labels = get_probe_labels(instances, "payoff_optimal")
    valence_labels = get_probe_labels(instances, "label_valence")

    assert list(payoff_labels) == [0, 0, 1, 1]
    assert list(valence_labels) == [0, 1, 1, 0]


def test_model_decision_is_passthrough():
    instances = [_FakeInst("A", "canon")] * 4
    pred = np.array([1, 0, 1, 1])
    decision_labels = get_probe_labels(instances, "model_decision", model_pred_slot=pred)
    assert list(decision_labels) == [1, 0, 1, 1]


def test_model_decision_requires_pred_slot():
    import pytest
    with pytest.raises(AssertionError):
        get_probe_labels([_FakeInst("A", "canon")], "model_decision")


def test_unknown_target_raises():
    import pytest
    with pytest.raises(ValueError):
        get_probe_labels([_FakeInst("A", "canon")], "not_a_real_target")
