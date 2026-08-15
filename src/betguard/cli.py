"""Command-line interface for BetGuard."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from betguard.artifact import (
    artifact_price_report,
    import_artifact,
    verify_artifact_directory,
    write_json,
)
from betguard.assurance import BetProposal, GateConfig, evaluate_proposal
from betguard.conditioning import analyze_design_matrix
from betguard.demo import run_demo
from betguard.formation import FormationThresholds, fit_affine_formation
from betguard.historical import (
    RECORD_TYPES,
    UserExportProvider,
    data_gap_report,
    import_historical_export,
)
from betguard.simulation import (
    PaperOutcome,
    add_paper_trade,
    analyze_paper_trades,
    load_paper_ledger,
    paper_trade_from_gate,
    save_paper_ledger,
    settle_paper_trade,
)
from betguard.video_pricing import (
    ControlledFormationThresholds,
    VideoPricingCalibration,
    analyze_team_payload,
    price_game_from_video,
)


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

    place_parser = subparsers.add_parser(
        "paper-place",
        help="gate and record a hypothetical trade; never places a real wager",
    )
    place_parser.add_argument("json_path", type=Path)
    place_parser.add_argument("ledger_path", type=Path)

    settle_parser = subparsers.add_parser(
        "paper-settle", help="settle a hypothetical trade in a paper ledger"
    )
    settle_parser.add_argument("ledger_path", type=Path)
    settle_parser.add_argument("trade_id")
    settle_parser.add_argument(
        "outcome",
        choices=[item.value for item in PaperOutcome if item is not PaperOutcome.OPEN],
    )
    settle_parser.add_argument("--settled-at", required=True)
    settle_parser.add_argument("--closing-decimal-odds", type=float)

    report_parser = subparsers.add_parser(
        "paper-report", help="summarize a hypothetical paper-trading ledger"
    )
    report_parser.add_argument("ledger_path", type=Path)

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

    video_parser = subparsers.add_parser(
        "video-price",
        help="price a simulated moneyline from audited, annotated video coordinates",
    )
    video_parser.add_argument("json_path", type=Path)

    artifact_import = subparsers.add_parser(
        "artifact-import",
        help="import a basketball affine-conditioning artifact directory",
    )
    artifact_import.add_argument("artifact_directory", type=Path)
    artifact_import.add_argument("--output", type=Path, required=True)
    artifact_import.add_argument(
        "--demo-two-possessions",
        action="store_true",
        help="exercise the two-possession fixture; never grants production qualification",
    )

    artifact_verify = subparsers.add_parser(
        "artifact-verify",
        help="verify compact hashes or stream the full telemetry CSV",
    )
    artifact_verify.add_argument("artifact_directory", type=Path)
    artifact_verify.add_argument("--stream-telemetry", action="store_true")
    artifact_verify.add_argument("--verify-large-hashes", action="store_true")

    artifact_price = subparsers.add_parser(
        "artifact-price",
        help="audit the artifact's synthetic scenario without treating arms as teams",
    )
    artifact_price.add_argument("artifact_directory", type=Path)
    artifact_price.add_argument("--home-decimal-odds", type=float)
    artifact_price.add_argument("--away-decimal-odds", type=float)
    artifact_price.add_argument("--output", type=Path)
    artifact_price.add_argument("--demo-two-possessions", action="store_true")

    historical_import = subparsers.add_parser(
        "historical-import",
        help="validate a user-supplied canonical historical export",
    )
    historical_import.add_argument("record_type", choices=sorted(RECORD_TYPES))
    historical_import.add_argument("source_path", type=Path)
    historical_import.add_argument("--output", type=Path, required=True)
    historical_import.add_argument("--provider-name", default="user_export")

    gap_report = subparsers.add_parser(
        "data-gap-report",
        help="emit the current machine-readable historical-data gap",
    )
    gap_report.add_argument("--output", type=Path)
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
    if args.command == "paper-place":
        payload = json.loads(args.json_path.read_text(encoding="utf-8"))
        result = _evaluate_payload(payload)
        output: dict[str, Any] = {
            "mode": "paper_only",
            "recorded": False,
            "assurance": result.to_dict(),
        }
        if result.approved_stake_fraction <= 0:
            print(json.dumps(output, indent=2))
            return 2
        proposal = BetProposal(**dict(payload["proposal"]))
        trade = paper_trade_from_gate(proposal, result, dict(payload["simulation"]))
        trades = add_paper_trade(load_paper_ledger(args.ledger_path), trade)
        save_paper_ledger(args.ledger_path, trades)
        output["recorded"] = True
        output["trade"] = trade.to_dict()
        print(json.dumps(output, indent=2))
        return 0
    if args.command == "paper-settle":
        outcome = PaperOutcome(args.outcome)
        if outcome is PaperOutcome.OPEN:
            raise ValueError("paper-settle outcome cannot be open")
        trades = settle_paper_trade(
            load_paper_ledger(args.ledger_path),
            args.trade_id,
            outcome,
            settled_at=args.settled_at,
            closing_decimal_odds=args.closing_decimal_odds,
        )
        save_paper_ledger(args.ledger_path, trades)
        print(json.dumps(analyze_paper_trades(trades).to_dict(), indent=2))
        return 0
    if args.command == "paper-report":
        trades = load_paper_ledger(args.ledger_path)
        print(json.dumps(analyze_paper_trades(trades).to_dict(), indent=2))
        return 0
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
    if args.command == "video-price":
        payload = json.loads(args.json_path.read_text(encoding="utf-8"))
        thresholds = ControlledFormationThresholds(**payload.get("thresholds", {}))
        calibration = VideoPricingCalibration(**payload.get("calibration", {}))
        home = analyze_team_payload(payload["home"], thresholds)
        away = analyze_team_payload(payload["away"], thresholds)
        market = payload["market"]
        price = price_game_from_video(
            home,
            away,
            baseline_home_probability=float(payload["baseline_home_probability"]),
            market_home_decimal_odds=float(market["home_decimal_odds"]),
            market_away_decimal_odds=float(market["away_decimal_odds"]),
            calibration=calibration,
            min_conservative_edge=float(payload.get("min_conservative_edge", 0.03)),
        )
        output = {
            "game": payload.get("game", "unspecified simulation"),
            "home_formation": home.to_dict(include_frames=False),
            "away_formation": away.to_dict(include_frames=False),
            "price": price.to_dict(),
        }
        print(json.dumps(output, indent=2))
        return 0
    if args.command == "artifact-import":
        report = import_artifact(
            args.artifact_directory,
            demo_two_possessions=args.demo_two_possessions,
        )
        payload = report.to_dict()
        write_json(args.output, payload)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "artifact-verify":
        report = verify_artifact_directory(
            args.artifact_directory,
            stream_telemetry=args.stream_telemetry,
            verify_large_hashes=args.verify_large_hashes,
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.passed else 2
    if args.command == "artifact-price":
        payload = artifact_price_report(
            args.artifact_directory,
            home_decimal_odds=args.home_decimal_odds,
            away_decimal_odds=args.away_decimal_odds,
            demo_two_possessions=args.demo_two_possessions,
        )
        if args.output is not None:
            write_json(args.output, payload)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "historical-import":
        report = import_historical_export(
            args.source_path,
            args.output,
            record_type=args.record_type,
            provider=UserExportProvider(args.provider_name),
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "data-gap-report":
        payload = data_gap_report()
        if args.output is not None:
            write_json(args.output, payload)
        print(json.dumps(payload, indent=2))
        return 0
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
