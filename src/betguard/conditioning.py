"""Numerical conditioning diagnostics for predictive-model feature matrices."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class DesignConditioningReport:
    """Summary of the numerical health of a model design matrix."""

    n_observations: int
    n_features: int
    rank: int
    effective_rank: float
    condition_number: float
    min_singular_value: float
    max_singular_value: float
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["condition_number"] = _finite_or_text(self.condition_number)
        return payload


def analyze_design_matrix(
    matrix: ArrayLike,
    *,
    condition_limit: float = 1_000.0,
    min_singular_value_limit: float = 1e-8,
    constant_tolerance: float = 1e-12,
    standardize: bool = True,
) -> DesignConditioningReport:
    """Analyze rank and conditioning of a two-dimensional feature matrix.

    Columns are centered and scaled by default so diagnostics are not dominated
    merely by different units. Constant columns are always reported as unsafe.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("matrix must contain at least two rows and one feature")
    if not np.isfinite(values).all():
        raise ValueError("matrix contains NaN or infinite values")
    if condition_limit <= 1:
        raise ValueError("condition_limit must be greater than 1")
    if min_singular_value_limit < 0 or constant_tolerance < 0:
        raise ValueError("singular-value tolerances must be non-negative")

    n_observations, n_features = values.shape
    reasons: list[str] = []

    if standardize:
        centered = values - values.mean(axis=0)
        scales = centered.std(axis=0)
        constant_columns = np.flatnonzero(scales <= constant_tolerance)
        if constant_columns.size:
            labels = ", ".join(str(int(index)) for index in constant_columns)
            reasons.append(f"constant or near-constant feature columns: {labels}")
        safe_scales = np.where(scales <= constant_tolerance, 1.0, scales)
        working = centered / safe_scales
    else:
        working = values.copy()
        spans = np.ptp(working, axis=0)
        constant_columns = np.flatnonzero(spans <= constant_tolerance)
        if constant_columns.size:
            labels = ", ".join(str(int(index)) for index in constant_columns)
            reasons.append(f"constant or near-constant feature columns: {labels}")

    singular_values = np.linalg.svd(working, compute_uv=False)
    rank = int(np.linalg.matrix_rank(working))
    max_singular_value = float(singular_values[0]) if singular_values.size else 0.0

    if n_features > singular_values.size or rank < n_features:
        min_singular_value = 0.0
        condition_number = float("inf")
    else:
        min_singular_value = float(singular_values[-1])
        condition_number = (
            float(max_singular_value / min_singular_value)
            if min_singular_value > 0
            else float("inf")
        )

    effective_rank = _entropy_effective_rank(singular_values)

    if rank < n_features:
        reasons.append(f"feature rank lost: rank {rank} < {n_features}")
    if min_singular_value < min_singular_value_limit:
        reasons.append(
            "minimum singular value below limit: "
            f"{min_singular_value:.3g} < {min_singular_value_limit:.3g}"
        )
    if condition_number > condition_limit:
        reasons.append(
            f"condition number above limit: {condition_number:.3g} > {condition_limit:.3g}"
        )

    return DesignConditioningReport(
        n_observations=n_observations,
        n_features=n_features,
        rank=rank,
        effective_rank=effective_rank,
        condition_number=condition_number,
        min_singular_value=min_singular_value,
        max_singular_value=max_singular_value,
        accepted=not reasons,
        reasons=tuple(reasons),
    )


def _entropy_effective_rank(singular_values: np.ndarray) -> float:
    energy = np.square(singular_values)
    total = float(energy.sum())
    if total <= 0:
        return 0.0
    probabilities = energy[energy > 0] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.exp(entropy))


def _finite_or_text(value: float) -> float | str:
    return value if np.isfinite(value) else "infinity"
