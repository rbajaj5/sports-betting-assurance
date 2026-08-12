import numpy as np
import pytest

from betguard.conditioning import analyze_design_matrix


def test_accepts_well_conditioned_matrix() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(300, 5))

    report = analyze_design_matrix(matrix)

    assert report.accepted
    assert report.rank == 5
    assert report.condition_number < 2


def test_rejects_duplicate_feature() -> None:
    rng = np.random.default_rng(4)
    feature = rng.normal(size=200)
    matrix = np.column_stack([feature, feature, rng.normal(size=200)])

    report = analyze_design_matrix(matrix)

    assert not report.accepted
    assert report.rank == 2
    assert report.condition_number == float("inf")
    assert any("rank lost" in reason for reason in report.reasons)


def test_rejects_constant_feature() -> None:
    matrix = np.column_stack([np.arange(20), np.ones(20)])

    report = analyze_design_matrix(matrix)

    assert not report.accepted
    assert any("constant" in reason for reason in report.reasons)


def test_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        analyze_design_matrix([[1.0, 2.0], [3.0, np.nan]])
