import numpy as np
import pytest

from betguard.formation import (
    analyze_formation_sequence,
    basketball_templates,
    fit_affine_formation,
    fit_best_template,
)


def test_identity_formation_is_well_conditioned() -> None:
    reference = basketball_templates()["five_out"]

    fit = fit_affine_formation(reference, reference, template_name="five_out")

    assert not fit.collapsed
    assert fit.singular_values == pytest.approx((1.0, 1.0))
    assert fit.determinant == pytest.approx(1.0)
    assert fit.residual_rmse == pytest.approx(0.0, abs=1e-12)


def test_rotation_and_translation_preserve_conditioning() -> None:
    reference = basketball_templates()["five_out"]
    angle = np.deg2rad(27)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    observed = reference @ rotation + np.array([4.0, 11.0])

    fit = fit_affine_formation(reference, observed)

    assert not fit.collapsed
    assert fit.condition_number == pytest.approx(1.0)
    assert fit.area_scale == pytest.approx(1.0)


def test_directional_compression_is_detected_despite_zero_residual() -> None:
    reference = basketball_templates()["five_out"]
    observed = reference @ np.array([[1.0, 0.0], [0.0, 0.03]]) + [3.0, 15.0]

    fit = fit_affine_formation(reference, observed)

    assert fit.residual_rmse == pytest.approx(0.0, abs=1e-12)
    assert fit.collapsed
    assert fit.singular_values[1] == pytest.approx(0.03)
    assert fit.condition_number == pytest.approx(1 / 0.03)


def test_uniform_crowding_is_detected_even_when_condition_number_is_one() -> None:
    reference = basketball_templates()["five_out"]
    observed = reference * 0.10 + [0.0, 8.0]

    fit = fit_affine_formation(reference, observed)

    assert fit.condition_number == pytest.approx(1.0)
    assert fit.collapsed
    assert any("singular value" in reason for reason in fit.reasons)
    assert any("area scale" in reason for reason in fit.reasons)


def test_sequence_reports_collapse_frames() -> None:
    reference = basketball_templates()["five_out"]
    compressions = [1.0, 0.7, 0.3, 0.1, 0.05, 0.5, 1.0]
    frames = np.stack([reference @ np.array([[1.0, 0.0], [0.0, value]]) for value in compressions])

    report = analyze_formation_sequence(reference, frames, template_name="five_out")

    assert report.collapse_frames == (3, 4)
    assert report.minimum_singular_value == pytest.approx(0.05)


def test_template_permutation_search_recovers_shuffled_players() -> None:
    templates = basketball_templates()
    reference = templates["horns"]
    observed = reference[[2, 4, 0, 3, 1]] + [2.0, 4.0]

    fit = fit_best_template(
        {"horns": templates["horns"]},
        observed,
        allow_permutation=True,
    )

    assert fit.template_name == "horns"
    assert fit.residual_rmse == pytest.approx(0.0, abs=1e-12)
