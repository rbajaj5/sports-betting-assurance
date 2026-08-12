"""Affine formation analysis for basketball player-tracking coordinates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class FormationThresholds:
    """Collapse thresholds for a fitted affine formation map."""

    min_singular_value: float = 0.25
    max_condition_number: float = 6.0
    min_area_scale: float = 0.15
    min_pairwise_spacing: float = 3.0


@dataclass(frozen=True)
class FormationFit:
    """Affine map and spacing diagnostics for one tracking frame."""

    template_name: str
    linear_map: tuple[tuple[float, float], tuple[float, float]]
    translation: tuple[float, float]
    singular_values: tuple[float, float]
    condition_number: float
    determinant: float
    area_scale: float
    min_pairwise_spacing: float
    residual_rmse: float
    collapsed: bool
    reasons: tuple[str, ...]
    assignment: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        if not np.isfinite(self.condition_number):
            payload["condition_number"] = "infinity"
        return payload


@dataclass(frozen=True)
class FormationSequenceReport:
    """Temporal summary of formation conditioning across tracking frames."""

    n_frames: int
    collapse_frames: tuple[int, ...]
    collapse_fraction: float
    minimum_singular_value: float
    maximum_condition_number: float
    minimum_area_scale: float
    minimum_pairwise_spacing: float
    mean_residual_rmse: float
    fits: tuple[FormationFit, ...]

    def to_dict(self, *, include_fits: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload: dict[str, Any] = {
            "n_frames": self.n_frames,
            "collapse_frames": self.collapse_frames,
            "collapse_fraction": self.collapse_fraction,
            "minimum_singular_value": self.minimum_singular_value,
            "maximum_condition_number": (
                self.maximum_condition_number
                if np.isfinite(self.maximum_condition_number)
                else "infinity"
            ),
            "minimum_area_scale": self.minimum_area_scale,
            "minimum_pairwise_spacing": self.minimum_pairwise_spacing,
            "mean_residual_rmse": self.mean_residual_rmse,
        }
        if include_fits:
            payload["fits"] = [fit.to_dict() for fit in self.fits]
        return payload


def basketball_templates() -> dict[str, np.ndarray]:
    """Return small half-court template library in feet.

    The basket is at ``(0, 0)``. The first coordinate runs sideline-to-sideline
    and the second runs away from the baseline. Rows represent stable roles and
    must be aligned with observed-player rows unless permutation search is used.
    """

    return {
        "five_out": np.array(
            [
                [0.0, 27.0],
                [-18.0, 19.0],
                [18.0, 19.0],
                [-22.0, 5.0],
                [22.0, 5.0],
            ]
        ),
        "horns": np.array(
            [
                [0.0, 28.0],
                [-8.0, 15.0],
                [8.0, 15.0],
                [-22.0, 5.0],
                [22.0, 5.0],
            ]
        ),
        "four_out_one_in": np.array(
            [
                [0.0, 27.0],
                [-18.0, 18.0],
                [18.0, 18.0],
                [-22.0, 5.0],
                [0.0, 6.0],
            ]
        ),
    }


def fit_affine_formation(
    reference_positions: ArrayLike,
    observed_positions: ArrayLike,
    *,
    template_name: str = "custom",
    thresholds: FormationThresholds | None = None,
    assignment: Iterable[int] | None = None,
) -> FormationFit:
    """Fit ``observed ~= reference @ A + translation`` for one frame.

    The singular values of ``A`` describe principal stretch. A small minimum
    singular value or determinant indicates collapse even when tracking residual
    is low. Minimum player spacing is checked separately because uniform
    compression has a condition number near one.
    """

    reference = _validate_positions(reference_positions, "reference_positions")
    observed = _validate_positions(observed_positions, "observed_positions")
    if reference.shape != observed.shape:
        raise ValueError("reference and observed positions must have the same shape")
    if reference.shape[0] < 3:
        raise ValueError("at least three player positions are required")

    design = np.column_stack([reference, np.ones(reference.shape[0])])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("reference positions must contain at least three non-collinear points")

    coefficients, _, _, _ = np.linalg.lstsq(design, observed, rcond=None)
    linear_map = coefficients[:2, :]
    translation = coefficients[2, :]
    predicted = design @ coefficients
    residual_rmse = float(np.sqrt(np.mean(np.sum(np.square(predicted - observed), axis=1))))

    singular_values = np.linalg.svd(linear_map, compute_uv=False)
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    condition_number = largest / smallest if smallest > 0 else float("inf")
    determinant = float(np.linalg.det(linear_map))
    area_scale = abs(determinant)
    min_spacing = _minimum_pairwise_distance(observed)
    limits = thresholds or FormationThresholds()
    _validate_thresholds(limits)

    reasons: list[str] = []
    if smallest < limits.min_singular_value:
        reasons.append(
            f"minimum singular value below limit: {smallest:.3g} < {limits.min_singular_value:.3g}"
        )
    if condition_number > limits.max_condition_number:
        reasons.append(
            f"condition number above limit: {condition_number:.3g} > "
            f"{limits.max_condition_number:.3g}"
        )
    if area_scale < limits.min_area_scale:
        reasons.append(
            f"formation area scale below limit: {area_scale:.3g} < {limits.min_area_scale:.3g}"
        )
    if min_spacing < limits.min_pairwise_spacing:
        reasons.append(
            f"minimum player spacing below limit: {min_spacing:.3g} < "
            f"{limits.min_pairwise_spacing:.3g}"
        )

    assigned = tuple(assignment) if assignment is not None else tuple(range(reference.shape[0]))
    return FormationFit(
        template_name=template_name,
        linear_map=_matrix_tuple(linear_map),
        translation=(float(translation[0]), float(translation[1])),
        singular_values=(largest, smallest),
        condition_number=condition_number,
        determinant=determinant,
        area_scale=area_scale,
        min_pairwise_spacing=min_spacing,
        residual_rmse=residual_rmse,
        collapsed=bool(reasons),
        reasons=tuple(reasons),
        assignment=assigned,
    )


def fit_best_template(
    templates: Mapping[str, ArrayLike],
    observed_positions: ArrayLike,
    *,
    thresholds: FormationThresholds | None = None,
    allow_permutation: bool = False,
) -> FormationFit:
    """Return the template/assignment with the lowest affine residual.

    Permutation search is practical for five-player frames (at most 120
    assignments). When reliable player roles are available, keeping their fixed
    row ordering is usually preferable.
    """

    if not templates:
        raise ValueError("at least one template is required")
    observed = _validate_positions(observed_positions, "observed_positions")
    best: FormationFit | None = None

    for name, template_positions in templates.items():
        reference = _validate_positions(template_positions, f"template '{name}'")
        if reference.shape != observed.shape:
            continue
        assignments: Iterable[tuple[int, ...]]
        if allow_permutation:
            assignments = permutations(range(observed.shape[0]))
        else:
            assignments = [tuple(range(observed.shape[0]))]
        for assignment in assignments:
            ordered_observed = observed[list(assignment)]
            candidate = fit_affine_formation(
                reference,
                ordered_observed,
                template_name=name,
                thresholds=thresholds,
                assignment=assignment,
            )
            if best is None or candidate.residual_rmse < best.residual_rmse:
                best = candidate

    if best is None:
        raise ValueError("no template has the same number of positions as the observed frame")
    return best


def analyze_formation_sequence(
    reference_positions: ArrayLike,
    frames: ArrayLike,
    *,
    template_name: str = "custom",
    thresholds: FormationThresholds | None = None,
) -> FormationSequenceReport:
    """Fit an ordered sequence of tracking frames against one reference shape."""

    reference = _validate_positions(reference_positions, "reference_positions")
    values = np.asarray(frames, dtype=float)
    if values.ndim != 3 or values.shape[1:] != reference.shape:
        raise ValueError("frames must have shape (n_frames, n_players, 2)")
    if values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("frames must be non-empty and finite")

    fits = tuple(
        fit_affine_formation(
            reference,
            frame,
            template_name=template_name,
            thresholds=thresholds,
        )
        for frame in values
    )
    collapse_frames = tuple(index for index, fit in enumerate(fits) if fit.collapsed)
    condition_numbers = [fit.condition_number for fit in fits]
    return FormationSequenceReport(
        n_frames=len(fits),
        collapse_frames=collapse_frames,
        collapse_fraction=len(collapse_frames) / len(fits),
        minimum_singular_value=min(fit.singular_values[1] for fit in fits),
        maximum_condition_number=max(condition_numbers),
        minimum_area_scale=min(fit.area_scale for fit in fits),
        minimum_pairwise_spacing=min(fit.min_pairwise_spacing for fit in fits),
        mean_residual_rmse=float(np.mean([fit.residual_rmse for fit in fits])),
        fits=fits,
    )


def _validate_positions(positions: ArrayLike, label: str) -> np.ndarray:
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{label} must have shape (n_players, 2)")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return values


def _minimum_pairwise_distance(positions: np.ndarray) -> float:
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt(np.sum(np.square(differences), axis=2))
    distances[np.diag_indices_from(distances)] = np.inf
    return float(np.min(distances))


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(matrix[0, 0]), float(matrix[0, 1])),
        (float(matrix[1, 0]), float(matrix[1, 1])),
    )


def _validate_thresholds(thresholds: FormationThresholds) -> None:
    if thresholds.min_singular_value < 0 or thresholds.min_area_scale < 0:
        raise ValueError("singular-value and area thresholds cannot be negative")
    if thresholds.max_condition_number <= 1:
        raise ValueError("max_condition_number must be greater than 1")
    if thresholds.min_pairwise_spacing < 0:
        raise ValueError("min_pairwise_spacing cannot be negative")
