import numpy as np
import pytest

from betguard.portfolio import (
    analyze_covariance,
    effective_number_of_bets,
    fractional_kelly_binary,
    shrink_covariance,
)


def test_independent_equal_risk_bets_have_full_effective_count() -> None:
    covariance = np.eye(4)

    assert effective_number_of_bets(covariance) == pytest.approx(4.0)
    assert analyze_covariance(covariance).accepted


def test_perfectly_correlated_bets_are_rejected() -> None:
    covariance = np.ones((3, 3))

    report = analyze_covariance(covariance)

    assert not report.accepted
    assert report.rank == 1
    assert report.effective_bets == pytest.approx(1.0)


def test_shrinkage_improves_rank_and_conditioning() -> None:
    covariance = np.ones((3, 3))

    shrunk = shrink_covariance(covariance, intensity=0.25)
    report = analyze_covariance(shrunk)

    assert report.rank == 3
    assert np.isfinite(report.condition_number)


def test_fractional_kelly_is_capped_and_never_negative() -> None:
    assert fractional_kelly_binary(0.60, 2.0, fraction=0.25, cap=0.01) == 0.01
    assert fractional_kelly_binary(0.40, 2.0) == 0.0
