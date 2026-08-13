"""Auditable, video-derived affine formation analysis and simulation pricing.

This module intentionally starts from annotated court coordinates rather than
claiming to recover reliable player tracking from arbitrary broadcast video.
The reference animation's runtime-assurance rule is preserved: use three
identified leaders, reject a poorly conditioned affine map, and hold the last
well-conditioned map while measuring follower deviation from it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

_RIGHTS_BASES = {"user_supplied", "official_public", "licensed", "synthetic"}


@dataclass(frozen=True)
class VideoEvidence:
    """Provenance and annotation quality for one team's video sample."""

    source_uri: str
    captured_at: str
    clip_sha256: str
    rights_basis: str
    annotation_method: str
    frame_timestamps: tuple[float, ...]
    possession_count: int
    lineup_coverage: float
    court_calibration_rmse_feet: float
    identities_verified: bool

    def __post_init__(self) -> None:
        if not self.source_uri.strip() or not self.annotation_method.strip():
            raise ValueError("video source and annotation method are required")
        try:
            datetime.fromisoformat(self.captured_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("captured_at must be an ISO-8601 timestamp") from exc
        digest = self.clip_sha256.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("clip_sha256 must be a 64-character hexadecimal digest")
        if self.rights_basis not in _RIGHTS_BASES:
            raise ValueError(f"rights_basis must be one of {sorted(_RIGHTS_BASES)}")
        if not self.frame_timestamps or any(value < 0 for value in self.frame_timestamps):
            raise ValueError("frame_timestamps must contain non-negative values")
        if self.possession_count < 0:
            raise ValueError("possession_count cannot be negative")
        if not 0 <= self.lineup_coverage <= 1:
            raise ValueError("lineup_coverage must be between zero and one")
        if self.court_calibration_rmse_feet < 0:
            raise ValueError("court calibration error cannot be negative")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VideoEvidence:
        values = dict(payload)
        values["frame_timestamps"] = tuple(float(item) for item in values["frame_timestamps"])
        return cls(**values)


@dataclass(frozen=True)
class ControlledFormationThresholds:
    """Geometry and evidence gates for the controlled formation layer."""

    min_normalized_leader_conditioning: float = 0.12
    min_area_scale: float = 0.15
    min_pairwise_spacing_feet: float = 3.0
    max_follower_rmse_feet: float = 5.0
    max_calibration_rmse_feet: float = 1.5
    min_usable_frames: int = 12
    min_possessions: int = 5
    min_lineup_coverage: float = 0.80
    min_well_conditioned_fraction: float = 0.60
    min_usable_fraction: float = 0.60

    def __post_init__(self) -> None:
        if not 0 < self.min_normalized_leader_conditioning <= 1:
            raise ValueError("normalized leader conditioning must be in (0, 1]")
        if self.min_area_scale < 0 or self.min_pairwise_spacing_feet < 0:
            raise ValueError("area and spacing thresholds cannot be negative")
        if self.max_follower_rmse_feet <= 0 or self.max_calibration_rmse_feet < 0:
            raise ValueError("RMSE limits must be positive or zero as appropriate")
        if self.min_usable_frames < 1 or self.min_possessions < 1:
            raise ValueError("minimum frame and possession counts must be positive")
        for name in ("min_lineup_coverage", "min_well_conditioned_fraction", "min_usable_fraction"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True)
class ControlledFormationFrame:
    """Runtime-assurance diagnostics for one annotated video frame."""

    frame_index: int
    timestamp_seconds: float
    normalized_leader_conditioning: float
    area_scale: float
    min_pairwise_spacing_feet: float
    follower_rmse_feet: float | None
    accepted_current_map: bool
    held_last_well_conditioned_map: bool
    usable: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlledFormationReport:
    """Team-level summary of the video-derived formation signal."""

    leader_ids: tuple[str, str, str]
    follower_ids: tuple[str, ...]
    n_frames: int
    accepted_frames: int
    held_frames: int
    rejected_frames: int
    well_conditioned_fraction: float
    usable_fraction: float
    median_normalized_leader_conditioning: float
    median_follower_rmse_feet: float | None
    top_three_impact_sum: float
    quality_qualified: bool
    quality_reasons: tuple[str, ...]
    evidence: VideoEvidence
    frames: tuple[ControlledFormationFrame, ...]

    def to_dict(self, *, include_frames: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_frames:
            payload.pop("frames")
        return payload


@dataclass(frozen=True)
class VideoPricingCalibration:
    """Explicit coefficients learned outside this module and supplied by the caller.

    Setting ``validated`` asserts that the coefficients were frozen and checked
    on out-of-sample historical games. The default remains an illustrative
    simulation and therefore can never qualify a paper candidate.
    """

    version: str = "illustrative-v1"
    historical_games: int = 0
    minimum_historical_games: int = 200
    validated: bool = False
    well_conditioned_logit_weight: float = 0.35
    leader_conditioning_logit_weight: float = 0.45
    follower_rmse_logit_weight: float = 0.20
    top_three_impact_logit_weight: float = 0.20
    follower_rmse_scale_feet: float = 5.0
    player_impact_scale: float = 10.0
    probability_uncertainty: float = 0.025
    max_abs_logit_adjustment: float = 0.75

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("calibration version is required")
        if self.historical_games < 0 or self.minimum_historical_games < 1:
            raise ValueError("historical sample sizes are invalid")
        if (
            self.follower_rmse_scale_feet <= 0
            or self.player_impact_scale <= 0
            or self.max_abs_logit_adjustment <= 0
        ):
            raise ValueError("pricing scales must be positive")
        if not 0 <= self.probability_uncertainty < 0.5:
            raise ValueError("probability_uncertainty must be in [0, 0.5)")

    @property
    def calibration_qualified(self) -> bool:
        return self.validated and self.historical_games >= self.minimum_historical_games


@dataclass(frozen=True)
class VideoGamePrice:
    """Simulation-only moneyline price derived from two formation reports."""

    mode: str
    formation_adjustment_applied: bool
    baseline_home_probability: float
    adjusted_home_probability: float
    adjusted_away_probability: float
    fair_home_decimal_odds: float
    fair_away_decimal_odds: float
    market_no_vig_home_probability: float
    market_no_vig_away_probability: float
    conservative_home_edge: float
    conservative_away_edge: float
    feature_deltas: dict[str, float]
    logit_contributions: dict[str, float]
    total_logit_adjustment: float
    probability_uncertainty: float
    scenario_preference: str | None
    paper_candidate: str | None
    qualified: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_controlled_formation(
    reference_positions: ArrayLike,
    frames: ArrayLike,
    *,
    player_ids: tuple[str, ...] | list[str],
    impact_ratings: dict[str, float],
    evidence: VideoEvidence,
    thresholds: ControlledFormationThresholds | None = None,
) -> ControlledFormationReport:
    """Analyze identified players using the top three impact ratings as leaders."""

    limits = thresholds or ControlledFormationThresholds()
    reference = _positions(reference_positions, "reference_positions")
    observed = np.asarray(frames, dtype=float)
    ids = tuple(player_ids)
    if observed.ndim != 3 or observed.shape[1:] != reference.shape:
        raise ValueError("frames must have shape (n_frames, n_players, 2)")
    if observed.shape[0] < 1 or not np.isfinite(observed).all():
        raise ValueError("frames must be non-empty and finite")
    if len(ids) != reference.shape[0] or len(ids) < 4 or len(set(ids)) != len(ids):
        raise ValueError("player_ids must uniquely identify at least four reference rows")
    if len(evidence.frame_timestamps) != observed.shape[0]:
        raise ValueError("evidence timestamps must align one-to-one with frames")
    missing = [player_id for player_id in ids if player_id not in impact_ratings]
    if missing:
        raise ValueError(f"impact ratings missing for: {', '.join(missing)}")
    ratings = {player_id: float(impact_ratings[player_id]) for player_id in ids}
    if not all(math.isfinite(value) for value in ratings.values()):
        raise ValueError("impact ratings must be finite")

    leader_ids = tuple(sorted(ids, key=lambda item: (-ratings[item], item))[:3])
    leader_indices = tuple(ids.index(player_id) for player_id in leader_ids)
    follower_indices = tuple(index for index in range(len(ids)) if index not in leader_indices)
    follower_ids = tuple(ids[index] for index in follower_indices)
    leader_reference = reference[list(leader_indices)]
    design = np.column_stack([leader_reference, np.ones(3)])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("the three selected leader reference positions must be non-collinear")

    last_well_conditioned: np.ndarray | None = None
    frame_reports: list[ControlledFormationFrame] = []
    for frame_index, frame in enumerate(observed):
        coefficients, _, _, _ = np.linalg.lstsq(
            design,
            frame[list(leader_indices)],
            rcond=None,
        )
        linear_map = coefficients[:2, :]
        singular_values = np.linalg.svd(linear_map, compute_uv=False)
        largest = float(singular_values[0])
        smallest = float(singular_values[-1])
        normalized_conditioning = smallest / largest if largest > 0 else 0.0
        area_scale = abs(float(np.linalg.det(linear_map)))
        spacing = _minimum_pairwise_distance(frame)

        geometry_reasons: list[str] = []
        if normalized_conditioning < limits.min_normalized_leader_conditioning:
            geometry_reasons.append(
                "normalized leader conditioning below limit: "
                f"{normalized_conditioning:.3g} < "
                f"{limits.min_normalized_leader_conditioning:.3g}"
            )
        if area_scale < limits.min_area_scale:
            geometry_reasons.append(
                f"formation area scale below limit: {area_scale:.3g} < {limits.min_area_scale:.3g}"
            )
        if spacing < limits.min_pairwise_spacing_feet:
            geometry_reasons.append(
                f"minimum player spacing below limit: {spacing:.3g} < "
                f"{limits.min_pairwise_spacing_feet:.3g}"
            )

        accepted_current = not geometry_reasons
        if accepted_current:
            last_well_conditioned = coefficients.copy()
        held = not accepted_current and last_well_conditioned is not None
        control_map = coefficients if accepted_current else last_well_conditioned
        follower_rmse: float | None = None
        reasons = list(geometry_reasons)
        if control_map is None:
            reasons.append("no prior well-conditioned affine map is available")
        else:
            follower_reference = reference[list(follower_indices)]
            follower_design = np.column_stack(
                [follower_reference, np.ones(len(follower_indices))]
            )
            follower_prediction = follower_design @ control_map
            follower_actual = frame[list(follower_indices)]
            follower_rmse = float(
                np.sqrt(np.mean(np.sum(np.square(follower_prediction - follower_actual), axis=1)))
            )
            if follower_rmse > limits.max_follower_rmse_feet:
                reasons.append(
                    f"follower RMSE above limit: {follower_rmse:.3g} > "
                    f"{limits.max_follower_rmse_feet:.3g} feet"
                )
        usable = control_map is not None and follower_rmse is not None and (
            follower_rmse <= limits.max_follower_rmse_feet
        )
        frame_reports.append(
            ControlledFormationFrame(
                frame_index=frame_index,
                timestamp_seconds=evidence.frame_timestamps[frame_index],
                normalized_leader_conditioning=normalized_conditioning,
                area_scale=area_scale,
                min_pairwise_spacing_feet=spacing,
                follower_rmse_feet=follower_rmse,
                accepted_current_map=accepted_current,
                held_last_well_conditioned_map=held,
                usable=usable,
                reasons=tuple(reasons),
            )
        )

    n_frames = len(frame_reports)
    accepted_frames = sum(frame.accepted_current_map for frame in frame_reports)
    held_frames = sum(frame.held_last_well_conditioned_map for frame in frame_reports)
    rejected_frames = n_frames - accepted_frames
    usable_frames = sum(frame.usable for frame in frame_reports)
    follower_values = [
        frame.follower_rmse_feet
        for frame in frame_reports
        if frame.follower_rmse_feet is not None
    ]
    well_conditioned_fraction = accepted_frames / n_frames
    usable_fraction = usable_frames / n_frames

    quality_reasons: list[str] = []
    if not evidence.identities_verified:
        quality_reasons.append("player identities were not verified")
    if evidence.court_calibration_rmse_feet > limits.max_calibration_rmse_feet:
        quality_reasons.append("court calibration error exceeds the configured limit")
    if usable_frames < limits.min_usable_frames:
        quality_reasons.append("too few usable controlled frames")
    if evidence.possession_count < limits.min_possessions:
        quality_reasons.append("too few possessions")
    if evidence.lineup_coverage < limits.min_lineup_coverage:
        quality_reasons.append("projected-lineup coverage is too low")
    if well_conditioned_fraction < limits.min_well_conditioned_fraction:
        quality_reasons.append("well-conditioned frame fraction is too low")
    if usable_fraction < limits.min_usable_fraction:
        quality_reasons.append("usable controlled-frame fraction is too low")

    return ControlledFormationReport(
        leader_ids=leader_ids,  # type: ignore[arg-type]
        follower_ids=follower_ids,
        n_frames=n_frames,
        accepted_frames=accepted_frames,
        held_frames=held_frames,
        rejected_frames=rejected_frames,
        well_conditioned_fraction=well_conditioned_fraction,
        usable_fraction=usable_fraction,
        median_normalized_leader_conditioning=float(
            np.median([frame.normalized_leader_conditioning for frame in frame_reports])
        ),
        median_follower_rmse_feet=(
            float(np.median(follower_values)) if follower_values else None
        ),
        top_three_impact_sum=sum(ratings[player_id] for player_id in leader_ids),
        quality_qualified=not quality_reasons,
        quality_reasons=tuple(quality_reasons),
        evidence=evidence,
        frames=tuple(frame_reports),
    )


def price_game_from_video(
    home: ControlledFormationReport,
    away: ControlledFormationReport,
    *,
    baseline_home_probability: float,
    market_home_decimal_odds: float,
    market_away_decimal_odds: float,
    calibration: VideoPricingCalibration | None = None,
    min_conservative_edge: float = 0.03,
) -> VideoGamePrice:
    """Create a de-vigged, simulation-only moneyline price.

    Formation coefficients are applied only when both video reports clear their
    provenance and quality gates. A paper candidate additionally requires an
    explicitly validated historical calibration and a conservative edge.
    """

    fitted = calibration or VideoPricingCalibration()
    if not 0 < baseline_home_probability < 1:
        raise ValueError("baseline_home_probability must be between zero and one")
    if market_home_decimal_odds <= 1 or market_away_decimal_odds <= 1:
        raise ValueError("two-sided decimal market odds must both exceed one")
    if min_conservative_edge < 0:
        raise ValueError("min_conservative_edge cannot be negative")

    home_implied = 1 / market_home_decimal_odds
    away_implied = 1 / market_away_decimal_odds
    overround = home_implied + away_implied
    market_home = home_implied / overround
    market_away = away_implied / overround

    formation_ready = home.quality_qualified and away.quality_qualified
    feature_deltas = {
        "well_conditioned_fraction": (
            home.well_conditioned_fraction - away.well_conditioned_fraction
        ),
        "leader_conditioning": (
            home.median_normalized_leader_conditioning
            - away.median_normalized_leader_conditioning
        ),
        "follower_rmse_advantage": 0.0,
        "top_three_impact": float(
            np.clip(
                (home.top_three_impact_sum - away.top_three_impact_sum)
                / fitted.player_impact_scale,
                -1,
                1,
            )
        ),
    }
    if home.median_follower_rmse_feet is not None and away.median_follower_rmse_feet is not None:
        feature_deltas["follower_rmse_advantage"] = float(
            np.clip(
                (away.median_follower_rmse_feet - home.median_follower_rmse_feet)
                / fitted.follower_rmse_scale_feet,
                -1,
                1,
            )
        )
    weights = {
        "well_conditioned_fraction": fitted.well_conditioned_logit_weight,
        "leader_conditioning": fitted.leader_conditioning_logit_weight,
        "follower_rmse_advantage": fitted.follower_rmse_logit_weight,
        "top_three_impact": fitted.top_three_impact_logit_weight,
    }
    contributions = {
        name: (feature_deltas[name] * weight if formation_ready else 0.0)
        for name, weight in weights.items()
    }
    total_adjustment = float(
        np.clip(
            sum(contributions.values()),
            -fitted.max_abs_logit_adjustment,
            fitted.max_abs_logit_adjustment,
        )
    )
    baseline_logit = math.log(baseline_home_probability / (1 - baseline_home_probability))
    adjusted_home = 1 / (1 + math.exp(-(baseline_logit + total_adjustment)))
    adjusted_away = 1 - adjusted_home
    conservative_home_edge = adjusted_home - fitted.probability_uncertainty - market_home
    conservative_away_edge = adjusted_away - fitted.probability_uncertainty - market_away
    scenario_preference: str | None = None
    best_edge = max(conservative_home_edge, conservative_away_edge)
    if formation_ready and best_edge > 0:
        scenario_preference = "home" if conservative_home_edge >= conservative_away_edge else "away"

    reasons: list[str] = []
    if not home.quality_qualified:
        reasons.extend(f"home video: {reason}" for reason in home.quality_reasons)
    if not away.quality_qualified:
        reasons.extend(f"away video: {reason}" for reason in away.quality_reasons)
    if not fitted.validated:
        reasons.append("formation calibration is illustrative, not historically validated")
    elif fitted.historical_games < fitted.minimum_historical_games:
        reasons.append("formation calibration sample is below its minimum")
    if best_edge < min_conservative_edge:
        reasons.append(
            f"best conservative edge is below {min_conservative_edge:.1%}: {best_edge:.1%}"
        )

    qualified = formation_ready and fitted.calibration_qualified and (
        best_edge >= min_conservative_edge
    )
    paper_candidate = scenario_preference if qualified else None
    return VideoGamePrice(
        mode="simulation_only",
        formation_adjustment_applied=formation_ready,
        baseline_home_probability=baseline_home_probability,
        adjusted_home_probability=adjusted_home,
        adjusted_away_probability=adjusted_away,
        fair_home_decimal_odds=1 / adjusted_home,
        fair_away_decimal_odds=1 / adjusted_away,
        market_no_vig_home_probability=market_home,
        market_no_vig_away_probability=market_away,
        conservative_home_edge=conservative_home_edge,
        conservative_away_edge=conservative_away_edge,
        feature_deltas=feature_deltas,
        logit_contributions=contributions,
        total_logit_adjustment=total_adjustment,
        probability_uncertainty=fitted.probability_uncertainty,
        scenario_preference=scenario_preference,
        paper_candidate=paper_candidate,
        qualified=qualified,
        reasons=tuple(reasons),
    )


def analyze_team_payload(
    payload: dict[str, Any],
    thresholds: ControlledFormationThresholds | None = None,
) -> ControlledFormationReport:
    """Build a controlled formation report from the documented JSON schema."""

    return analyze_controlled_formation(
        payload["reference_positions"],
        payload["frames"],
        player_ids=payload["player_ids"],
        impact_ratings=payload["impact_ratings"],
        evidence=VideoEvidence.from_dict(payload["evidence"]),
        thresholds=thresholds,
    )


def _positions(values: ArrayLike, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{label} must have shape (n_players, 2)")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return array


def _minimum_pairwise_distance(positions: np.ndarray) -> float:
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt(np.sum(np.square(differences), axis=2))
    distances[np.diag_indices_from(distances)] = np.inf
    return float(np.min(distances))
