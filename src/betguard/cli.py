"""Command-line interface for BetGuard."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from betguard.assurance import BetProposal, GateConfig, evaluate_proposal
from betguard.conditioning import analyze_design_matrix
from betguard.demo import run_demo
from betguard.formation import FormationThresholds, fit_affine_formation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="betguard",
        description="Conditioning-aware assurance checks for betting systems",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("demo", help="run the synthetic basketball demonstration")

    design_parser = subparsers.add_parser(
        "check-design", help="check selected numeric columns in a CSV"
    )
    design_parser.add_argument("csv_path", type=Path)
    design_parser.add_argument(
        "--columns",
        required=True,
        help="comma-separated numeric feature columns",
    )
    design_parser.add_argument("--condition-limit", type=float, default=1_000.0)
    design_parser.add_argument("--min-singular-value", type=float, default=1e-8)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a proposal from a JSON file")
    evaluate_parser.add_argument("json_path", type=Path)

    formation_parser = subparsers.add_parser(
        "check-formation", help="fit one observed basketball formation to a reference"
    )
    formation_parser.add_argument("reference_json", type=Path)
    formation_parser.add_argument("observed_json", type=Path)
    formation_parser.add_argument("--template-name", default="custom")
    formation_parser.add_argument("--min-singular-value", type=float, default=0.25)
    formation_parser.add_argument("--max-condition-number", type=float, default=6.0)
    formation_parser.add_argument("--min-area-scale", type=float, default=0.15)
    formation_parser.add_argument("--min-spacing", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        run_demo()
        return 0
    if args.command == "check-design":
        columns = [column.strip() for column in args.columns.split(",") if column.strip()]
        matrix = _read_numeric_csv(args.csv_path, columns)
        report = analyze_design_matrix(
            matrix,
            condition_limit=args.condition_limit,
            min_singular_value_limit=args.min_singular_value,
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.accepted else 2
    if args.command == "evaluate":
        payload = json.loads(args.json_path.read_text(encoding="utf-8"))
        result = _evaluate_payload(payload)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.approved_stake_fraction > 0 else 2
    if args.command == "check-formation":
        reference = _read_positions_json(args.reference_json)
        observed = _read_positions_json(args.observed_json)
        thresholds = FormationThresholds(
            min_singular_value=args.min_singular_value,
            max_condition_number=args.max_condition_number,
            min_area_scale=args.min_area_scale,
            min_pairwise_spacing=args.min_spacing,
        )
        result = fit_affine_formation(
            reference,
            observed,
            template_name=args.template_name,
            thresholds=thresholds,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 2 if result.collapsed else 0
    raise AssertionError(f"unsupported command: {args.command}")


def _read_numeric_csv(path: Path, columns: list[str]) -> np.ndarray:
    if not columns:
        raise ValueError("at least one column is required")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
        rows = [[float(row[column]) for column in columns] for row in reader]
    if not rows:
        raise ValueError("CSV contains no data rows")
    return np.asarray(rows, dtype=float)


def _evaluate_payload(payload: dict[str, Any]):
    proposal_payload = dict(payload["proposal"])
    proposal = BetProposal(**proposal_payload)
    config = GateConfig(**payload.get("config", {}))
    return evaluate_proposal(
        proposal,
        config=config,
        current_factor_exposure=payload.get("current_factor_exposure", {}),
    )


def _read_positions_json(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    positions = payload["positions"] if isinstance(payload, dict) else payload
    return np.asarray(positions, dtype=float)


if __name__ == "__main__":
    raise SystemExit(main())
