"""Leakage-aware historical evaluation and fail-closed qualification gates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np

NO_TRADE_VERDICT = "PASS — no qualified trade."
FORBIDDEN_PREGAME_FIELDS = {
    "home_final_score",
    "away_final_score",
    "result",
    "postgame_status",
    "future_market_price",
}


@dataclass(frozen=True)
class ChronologicalSplit:
    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    test: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class QualificationInputs:
    verified_real_game_provenance: bool
    stable_player_identities: bool
    calibration_rmse_feet: float
    usable_frames: int
    possessions: int
    projected_lineup_coverage: float
    pregame_impact_ratings_present: bool
    availability_assumptions_present: bool
    coefficients_version: str | None
    coefficients_frozen: bool
    qualifying_historical_games: int
    regulated_two_sided_prices: bool
    conservative_post_uncertainty_edge: float
    synthetic_or_demo_override: bool = False


@dataclass(frozen=True)
class QualificationResult:
    qualified: bool
    reasons: tuple[str, ...]
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbabilityMetrics:
    count: int
    brier_score: float
    log_loss: float
    calibration_intercept: float
    calibration_slope: float
    reliability_bins: tuple[dict[str, float | int], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenLogisticModel:
    """A fitted coefficient vector whose preprocessing is part of the artifact."""

    version: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    training_game_ids: tuple[str, ...]
    layer: str
    qualifying_real_games: int

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("prediction features do not match the frozen feature contract")
        standardized = (matrix - np.asarray(self.means)) / np.asarray(self.scales)
        design = np.column_stack([np.ones(len(matrix)), standardized])
        linear = design @ np.asarray(self.coefficients)
        return 1 / (1 + np.exp(-np.clip(linear, -35, 35)))


@dataclass(frozen=True)
class FrozenFormationIncrement:
    """Formation log-odds increment fitted after a baseline is frozen."""

    version: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    training_game_ids: tuple[str, ...]
    qualifying_real_games: int
    baseline_version: str

    def adjust(
        self,
        baseline_probabilities: Sequence[float],
        formation_features: Sequence[Sequence[float]],
    ) -> np.ndarray:
        baseline = np.asarray(baseline_probabilities, dtype=float)
        matrix = np.asarray(formation_features, dtype=float)
        if matrix.ndim != 2 or matrix.shape != (len(baseline), len(self.feature_names)):
            raise ValueError("formation features do not match the frozen increment")
        if np.any((baseline <= 0) | (baseline >= 1)):
            raise ValueError("baseline probabilities must be in (0, 1)")
        standardized = (matrix - np.asarray(self.means)) / np.asarray(self.scales)
        design = np.column_stack([np.ones(len(matrix)), standardized])
        offset = np.log(baseline / (1 - baseline))
        linear = offset + design @ np.asarray(self.coefficients)
        return 1 / (1 + np.exp(-np.clip(linear, -35, 35)))


def devig_two_sided(decimal_odds_a: float, decimal_odds_b: float) -> tuple[float, float]:
    """Normalize two displayed implied probabilities to remove overround."""

    if decimal_odds_a <= 1 or decimal_odds_b <= 1:
        raise ValueError("two-sided decimal odds must both exceed one")
    implied_a = 1 / decimal_odds_a
    implied_b = 1 / decimal_odds_b
    total = implied_a + implied_b
    return implied_a / total, implied_b / total


def fit_pregame_baseline(
    features: Sequence[Sequence[float]],
    outcomes: Sequence[int],
    *,
    feature_names: Sequence[str],
    training_game_ids: Sequence[str],
    version: str,
    l2_penalty: float = 1e-3,
) -> FrozenLogisticModel:
    """Fit a reproducible baseline after callers pass the leakage checks."""

    matrix, observed = _model_arrays(features, outcomes, feature_names, training_game_ids)
    means, scales, standardized = _standardize(matrix)
    design = np.column_stack([np.ones(len(matrix)), standardized])
    coefficients = _fit_logistic_design(design, observed, l2_penalty=l2_penalty)
    return FrozenLogisticModel(
        version=_required_version(version),
        feature_names=tuple(feature_names),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
        training_game_ids=tuple(training_game_ids),
        layer="pregame_baseline",
        qualifying_real_games=len(matrix),
    )


def fit_formation_increment(
    baseline_probabilities: Sequence[float],
    formation_features: Sequence[Sequence[float]],
    outcomes: Sequence[int],
    *,
    feature_names: Sequence[str],
    training_game_ids: Sequence[str],
    version: str,
    baseline_version: str,
    verified_real_game_evidence: bool,
    minimum_games: int = 200,
    l2_penalty: float = 1e-3,
) -> FrozenFormationIncrement:
    """Fit residual log-odds only for a qualifying real-game sample."""

    if not verified_real_game_evidence:
        raise ValueError("synthetic formation evidence cannot fit outcome coefficients")
    matrix, observed = _model_arrays(
        formation_features,
        outcomes,
        feature_names,
        training_game_ids,
    )
    if len(matrix) < minimum_games:
        raise ValueError(
            f"formation calibration requires at least {minimum_games} qualifying real games"
        )
    baseline = np.asarray(baseline_probabilities, dtype=float)
    if baseline.shape != (len(matrix),) or np.any((baseline <= 0) | (baseline >= 1)):
        raise ValueError("baseline probabilities must align and lie in (0, 1)")
    means, scales, standardized = _standardize(matrix)
    design = np.column_stack([np.ones(len(matrix)), standardized])
    offsets = np.log(baseline / (1 - baseline))
    coefficients = _fit_logistic_design(
        design,
        observed,
        offsets=offsets,
        l2_penalty=l2_penalty,
    )
    return FrozenFormationIncrement(
        version=_required_version(version),
        feature_names=tuple(feature_names),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
        training_game_ids=tuple(training_game_ids),
        qualifying_real_games=len(matrix),
        baseline_version=_required_version(baseline_version),
    )


def chronological_split(
    records: Sequence[dict[str, Any]],
    *,
    timestamp_field: str,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    id_field: str = "game_id",
) -> ChronologicalSplit:
    """Create non-overlapping time-ordered partitions with stable game IDs."""

    if not records:
        raise ValueError("at least one historical record is required")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train plus validation must leave an untouched test period")
    copied = [dict(record) for record in records]
    identities = [str(record[id_field]) for record in copied]
    if len(set(identities)) != len(identities):
        raise ValueError("game IDs must be unique before chronological splitting")
    ordered = sorted(copied, key=lambda record: _timestamp(record[timestamp_field]))
    count = len(ordered)
    train_end = max(1, int(count * train_fraction))
    validation_end = max(train_end + 1, int(count * (train_fraction + validation_fraction)))
    if validation_end >= count:
        raise ValueError("too few records for non-empty train, validation, and test periods")
    split = ChronologicalSplit(
        train=tuple(ordered[:train_end]),
        validation=tuple(ordered[train_end:validation_end]),
        test=tuple(ordered[validation_end:]),
    )
    if _timestamp(split.train[-1][timestamp_field]) > _timestamp(
        split.validation[0][timestamp_field]
    ):
        raise AssertionError("training period overlaps validation")
    if _timestamp(split.validation[-1][timestamp_field]) > _timestamp(
        split.test[0][timestamp_field]
    ):
        raise AssertionError("validation period overlaps test")
    return split


def assert_pregame_row(
    row: dict[str, Any],
    *,
    feature_fields: Sequence[str],
    market_timestamp_field: str = "market_observation_timestamp",
    start_timestamp_field: str = "scheduled_start_time",
) -> None:
    """Reject postmarket/postgame information from a pregame feature row."""

    forbidden = FORBIDDEN_PREGAME_FIELDS.intersection(feature_fields)
    if forbidden:
        raise ValueError(f"pregame feature set contains forbidden fields: {sorted(forbidden)}")
    market_time = _timestamp(row[market_timestamp_field])
    start_time = _timestamp(row[start_timestamp_field])
    if market_time > start_time:
        raise ValueError("market observation occurs after scheduled game start")
    timestamps = row.get("feature_available_at", {})
    if not isinstance(timestamps, dict):
        raise ValueError("feature_available_at must map each feature to a timestamp")
    for field in feature_fields:
        if field not in row:
            raise ValueError(f"missing pregame feature: {field}")
        if field not in timestamps:
            raise ValueError(f"missing availability timestamp for feature: {field}")
        if _timestamp(timestamps[field]) > market_time:
            raise ValueError(f"feature became available after the quoted market: {field}")


def probability_metrics(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    *,
    bins: int = 10,
) -> ProbabilityMetrics:
    """Compute proper scores, logistic calibration, and reliability bins."""

    predicted, observed = _probability_arrays(probabilities, outcomes)
    clipped = np.clip(predicted, 1e-12, 1 - 1e-12)
    brier = float(np.mean(np.square(clipped - observed)))
    log_loss = float(
        -np.mean(observed * np.log(clipped) + (1 - observed) * np.log(1 - clipped))
    )
    intercept, slope = _calibration_fit(clipped, observed)
    edges = np.linspace(0, 1, bins + 1)
    rows: list[dict[str, float | int]] = []
    for index in range(bins):
        right_closed = index == bins - 1
        mask = (clipped >= edges[index]) & (
            clipped <= edges[index + 1] if right_closed else clipped < edges[index + 1]
        )
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "count": int(np.sum(mask)),
                "mean_prediction": float(np.mean(clipped[mask])),
                "observed_rate": float(np.mean(observed[mask])),
            }
        )
    return ProbabilityMetrics(
        count=len(clipped),
        brier_score=brier,
        log_loss=log_loss,
        calibration_intercept=intercept,
        calibration_slope=slope,
        reliability_bins=tuple(rows),
    )


def bootstrap_metric_interval(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    samples: int = 2_000,
    seed: int = 20260815,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval using game-level resampling."""

    predicted, observed = _probability_arrays(probabilities, outcomes)
    if samples < 100:
        raise ValueError("bootstrap requires at least 100 samples")
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(predicted), size=len(predicted))
        values[index] = metric(predicted[selected], observed[selected])
    return (
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )


def evaluate_qualification(inputs: QualificationInputs) -> QualificationResult:
    """Apply every production qualification gate without silent overrides."""

    reasons: list[str] = []
    if inputs.synthetic_or_demo_override:
        reasons.append("synthetic evidence or a demo override cannot qualify")
    if not inputs.verified_real_game_provenance:
        reasons.append("real-game provenance is not verified")
    if not inputs.stable_player_identities:
        reasons.append("player identities are not stable and verified")
    if inputs.calibration_rmse_feet > 1.5:
        reasons.append("court calibration RMSE exceeds 1.5 feet")
    if inputs.usable_frames < 12:
        reasons.append("fewer than 12 usable frames")
    if inputs.possessions < 5:
        reasons.append("fewer than five possessions")
    if inputs.projected_lineup_coverage < 0.80:
        reasons.append("projected-lineup coverage is below 80%")
    if not inputs.pregame_impact_ratings_present:
        reasons.append("pregame impact ratings are missing")
    if not inputs.availability_assumptions_present:
        reasons.append("availability assumptions are missing")
    if not inputs.coefficients_version or not inputs.coefficients_frozen:
        reasons.append("formation coefficients are not frozen and versioned")
    if inputs.qualifying_historical_games < 200:
        reasons.append("fewer than 200 qualifying historical games")
    if not inputs.regulated_two_sided_prices:
        reasons.append("current regulated two-sided prices are unavailable")
    if inputs.conservative_post_uncertainty_edge < 0.03:
        reasons.append("conservative post-uncertainty edge is below three percentage points")
    qualified = not reasons
    return QualificationResult(
        qualified=qualified,
        reasons=tuple(reasons),
        verdict="QUALIFIED PAPER CANDIDATE" if qualified else NO_TRADE_VERDICT,
    )


def flat_stake_paper_metrics(
    profits_in_units: Sequence[float],
    *,
    stake_units: float = 0.25,
    closing_line_values: Sequence[float] | None = None,
) -> dict[str, float | int | None]:
    """Summarize theoretical flat-stake paper returns without money execution."""

    if stake_units <= 0:
        raise ValueError("stake_units must be positive")
    profits = np.asarray(profits_in_units, dtype=float)
    if profits.ndim != 1 or not np.isfinite(profits).all():
        raise ValueError("paper profits must be a finite one-dimensional sequence")
    if len(profits) == 0:
        return {
            "bets": 0,
            "turnover_units": 0.0,
            "profit_units": 0.0,
            "roi": None,
            "maximum_drawdown_units": 0.0,
            "mean_closing_line_value": None,
        }
    equity = np.cumsum(profits)
    peaks = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    drawdown = peaks - equity
    clv = None
    if closing_line_values is not None:
        values = np.asarray(closing_line_values, dtype=float)
        if values.shape != profits.shape or not np.isfinite(values).all():
            raise ValueError("closing-line values must align with paper profits")
        clv = float(np.mean(values))
    turnover = len(profits) * stake_units
    return {
        "bets": len(profits),
        "turnover_units": turnover,
        "profit_units": float(np.sum(profits)),
        "roi": float(np.sum(profits) / turnover),
        "maximum_drawdown_units": float(np.max(drawdown)),
        "mean_closing_line_value": clv,
    }


def _probability_arrays(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(probabilities, dtype=float)
    observed = np.asarray(outcomes, dtype=float)
    if predicted.ndim != 1 or observed.shape != predicted.shape or len(predicted) == 0:
        raise ValueError("probabilities and outcomes must be aligned non-empty vectors")
    if not np.isfinite(predicted).all() or np.any((predicted <= 0) | (predicted >= 1)):
        raise ValueError("probabilities must be finite and strictly between zero and one")
    if not np.all(np.isin(observed, [0, 1])):
        raise ValueError("outcomes must be binary")
    return predicted, observed


def _calibration_fit(probabilities: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
    if np.all(outcomes == outcomes[0]):
        return float("nan"), float("nan")
    logits = np.log(probabilities / (1 - probabilities))
    design = np.column_stack([np.ones(len(logits)), logits])
    coefficients = np.array([0.0, 1.0], dtype=float)
    for _ in range(50):
        linear = design @ coefficients
        fitted = 1 / (1 + np.exp(-np.clip(linear, -35, 35)))
        weights = np.clip(fitted * (1 - fitted), 1e-9, None)
        gradient = design.T @ (outcomes - fitted)
        hessian = design.T @ (weights[:, None] * design)
        step = np.linalg.pinv(hessian) @ gradient
        coefficients += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(coefficients[0]), float(coefficients[1])


def _model_arrays(
    features: Sequence[Sequence[float]],
    outcomes: Sequence[int],
    feature_names: Sequence[str],
    game_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=float)
    observed = np.asarray(outcomes, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names) or matrix.shape[0] < 2:
        raise ValueError("feature matrix does not match its names")
    if observed.shape != (len(matrix),) or not np.all(np.isin(observed, [0, 1])):
        raise ValueError("outcomes must be aligned and binary")
    if len(game_ids) != len(matrix) or len(set(game_ids)) != len(game_ids):
        raise ValueError("training game IDs must be aligned and unique")
    if not np.isfinite(matrix).all():
        raise ValueError("model features must be finite")
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature names must be non-empty and unique")
    return matrix, observed


def _standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    if np.any(scales <= 1e-12):
        raise ValueError("model features contain a constant or near-constant column")
    return means, scales, (matrix - means) / scales


def _fit_logistic_design(
    design: np.ndarray,
    outcomes: np.ndarray,
    *,
    offsets: np.ndarray | None = None,
    l2_penalty: float,
) -> np.ndarray:
    if l2_penalty < 0:
        raise ValueError("l2_penalty cannot be negative")
    fixed_offset = np.zeros(len(design)) if offsets is None else offsets
    coefficients = np.zeros(design.shape[1], dtype=float)
    penalty = np.eye(design.shape[1]) * l2_penalty
    penalty[0, 0] = 0
    for _ in range(100):
        linear = fixed_offset + design @ coefficients
        fitted = 1 / (1 + np.exp(-np.clip(linear, -35, 35)))
        weights = np.clip(fitted * (1 - fitted), 1e-9, None)
        gradient = design.T @ (outcomes - fitted) - penalty @ coefficients
        hessian = design.T @ (weights[:, None] * design) + penalty
        step = np.linalg.pinv(hessian) @ gradient
        coefficients += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return coefficients


def _required_version(value: str) -> str:
    if not value.strip():
        raise ValueError("a non-empty frozen version is required")
    return value


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("chronological timestamps must carry a timezone")
    return parsed
