"""Portfolio concentration, covariance conditioning, and staking helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class PortfolioConditioningReport:
    """Summary of covariance stability and hidden portfolio concentration."""

    n_bets: int
    rank: int
    effective_bets: float
    condition_number: float
    min_eigenvalue: float
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        if not np.isfinite(self.condition_number):
            payload["condition_number"] = "infinity"
        return payload


def shrink_covariance(covariance: ArrayLike, intensity: float = 0.25) -> np.ndarray:
    """Shrink a covariance matrix toward its diagonal."""

    matrix = _validate_square_matrix(covariance)
    if not 0 <= intensity <= 1:
        raise ValueError("intensity must be between 0 and 1")
    diagonal_target = np.diag(np.diag(matrix))
    return (1 - intensity) * matrix + intensity * diagonal_target


def effective_number_of_bets(covariance: ArrayLike) -> float:
    """Return the eigenvalue participation ratio of a covariance matrix."""

    matrix = _validate_square_matrix(covariance)
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    squared_total = float(np.square(eigenvalues).sum())
    if total <= 0 or squared_total <= 0:
        return 0.0
    return total * total / squared_total


def analyze_covariance(
    covariance: ArrayLike,
    *,
    condition_limit: float = 10_000.0,
    min_effective_bets: float = 1.5,
    eigenvalue_tolerance: float = 1e-12,
) -> PortfolioConditioningReport:
    """Diagnose singularity and hidden concentration in bet-return covariance."""

    matrix = _validate_square_matrix(covariance)
    if condition_limit <= 1:
        raise ValueError("condition_limit must be greater than 1")
    if min_effective_bets < 1:
        raise ValueError("min_effective_bets must be at least 1")

    n_bets = matrix.shape[0]
    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eigenvalue = float(eigenvalues[0])
    max_eigenvalue = float(eigenvalues[-1])
    clipped = np.clip(eigenvalues, 0.0, None)
    rank = int(np.count_nonzero(clipped > eigenvalue_tolerance))
    reasons: list[str] = []

    if min_eigenvalue < -eigenvalue_tolerance:
        reasons.append("covariance matrix is not positive semidefinite")

    if rank < n_bets or clipped[0] <= eigenvalue_tolerance:
        condition_number = float("inf")
    else:
        condition_number = max_eigenvalue / float(clipped[0])

    effective_bets = effective_number_of_bets(matrix)
    if rank < n_bets:
        reasons.append(f"covariance rank lost: rank {rank} < {n_bets}")
    if condition_number > condition_limit:
        reasons.append(
            f"condition number above limit: {condition_number:.3g} > {condition_limit:.3g}"
        )
    if effective_bets < min_effective_bets:
        reasons.append(
            f"effective bets below limit: {effective_bets:.3g} < {min_effective_bets:.3g}"
        )

    return PortfolioConditioningReport(
        n_bets=n_bets,
        rank=rank,
        effective_bets=effective_bets,
        condition_number=condition_number,
        min_eigenvalue=min_eigenvalue,
        accepted=not reasons,
        reasons=tuple(reasons),
    )


def fractional_kelly_binary(
    probability: float,
    decimal_odds: float,
    *,
    fraction: float = 0.25,
    cap: float = 0.01,
) -> float:
    """Calculate a capped fractional-Kelly stake for a binary wager."""

    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if decimal_odds <= 1:
        raise ValueError("decimal_odds must be greater than 1")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    if cap < 0:
        raise ValueError("cap must be non-negative")

    net_odds = decimal_odds - 1
    full_kelly = (net_odds * probability - (1 - probability)) / net_odds
    return min(cap, max(0.0, full_kelly * fraction))


def _validate_square_matrix(matrix: ArrayLike) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 1:
        raise ValueError("covariance must be a non-empty square matrix")
    if not np.isfinite(values).all():
        raise ValueError("covariance contains NaN or infinite values")
    if not np.allclose(values, values.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    return (values + values.T) / 2
