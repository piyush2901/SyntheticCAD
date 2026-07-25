"""Command-line entry points for the SyntheticCAD core pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from syntheticcad.dashboard import write_executive_dashboard
from syntheticcad.mapping_guide import write_mapping_guide
from syntheticcad.privacy import build_privacy_report
from syntheticcad.profiling import build_profile, read_csv, write_json
from syntheticcad.sample_data import make_sample_cad
from syntheticcad.schema import SyntheticCADMapping, load_mapping, save_mapping
from syntheticcad.sensitive import field_profile
from syntheticcad.synthesis import (
    synthesize_conditional_result,
    synthesize_sdv,
    write_export_package,
)
from syntheticcad.tabular import synthesize_single_table
from syntheticcad.tabular_dashboard import write_tabular_dashboard
from syntheticcad.validation import validate_synthetic_data


def _cmd_make_sample(args: argparse.Namespace) -> int:
    path = make_sample_cad(args.out, seed=args.seed, event_count=args.events)
    print(f"Wrote sample CAD CSV: {path}")
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    df = read_csv(args.csv, nrows=args.max_rows)
    profile = build_profile(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = out_dir / "profile.json"
    mapping_path = out_dir / "mapping_suggested.json"
    guide_path = out_dir / "mapping_guide.html"

    write_json(profile, profile_path)
    save_mapping(SyntheticCADMapping.from_dict(profile["suggested_mapping"]), mapping_path)
    write_mapping_guide(profile, guide_path)

    diagnostics = profile["event_unit_diagnostics"]
    print(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")
    if args.max_rows:
        print(f"Profile limited to first {args.max_rows} source rows.")
    print(f"Wrote profile: {profile_path}")
    print(f"Wrote suggested mapping: {mapping_path}")
    print(f"Wrote mapping guide: {guide_path}")
    if diagnostics.get("available"):
        print(
            "Event/unit structure: "
            f"{diagnostics['event_count']} events, "
            f"{diagnostics['rows_per_event']['mean']} rows per event on average"
        )
    else:
        print(f"Event/unit structure unavailable: {diagnostics.get('reason')}")
    return 0


def _cmd_synthesize(args: argparse.Namespace) -> int:
    mapping = load_mapping(args.mapping)
    source_limit = args.source_max_rows
    source_header = read_csv(args.csv, nrows=0)
    source_column_count = int(source_header.shape[1])
    mapped_columns = mapping.mapped_columns()
    real_df = read_csv(
        args.csv,
        usecols=mapped_columns,
        nrows=source_limit,
    )
    if args.method == "conditional":
        synthesis = synthesize_conditional_result(
            real_df,
            mapping,
            event_count=args.events,
            seed=args.seed,
        )
    else:
        synthesis = synthesize_sdv(
            real_df,
            mapping,
            event_count=args.events,
            seed=args.seed,
        )
    synthetic_df = synthesis.dataframe
    validation = validate_synthetic_data(real_df, synthetic_df, mapping)
    run_metadata = {
        "source_file_name": Path(args.csv).name,
        "source_rows_used": int(real_df.shape[0]),
        "source_row_limit": source_limit,
        "source_columns_found": source_column_count,
        "mapped_columns_used": len(mapped_columns),
        "mapped_column_names": mapped_columns,
        "requested_synthetic_events": args.events,
        "synthetic_rows_created": int(synthetic_df.shape[0]),
        "synthesis_method": synthesis.method,
        "synthetic_generation_method": synthesis.method_summary,
        "synthesis_library": synthesis.library_used,
    }
    validation["methodology"] = {
        "library_used": synthesis.library_used,
        "method_summary": synthesis.method_summary,
        "offline_processing": True,
        "method": synthesis.method,
        "details": synthesis.details,
    }
    validation["generation_process"] = {
        "plain_language": (
            f"Read {real_df.shape[0]:,} real rows from {Path(args.csv).name}, "
            f"used the approved field mapping and {synthesis.library_used}, and generated "
            f"{synthetic_df.shape[0]:,} synthetic rows locally."
        ),
        **run_metadata,
    }
    privacy_report = build_privacy_report(real_df, synthetic_df, mapping, seed=args.seed)
    validation["privacy_risk"] = privacy_report
    paths = write_export_package(synthetic_df, args.out_dir, validation_report=validation)
    privacy_path = Path(args.out_dir) / "privacy_report.json"
    write_json(privacy_report, privacy_path)
    paths["privacy_report"] = privacy_path
    if synthesis.method.startswith("sdv") and synthesis.details.get("metadata"):
        metadata_path = Path(args.out_dir) / "sdv_metadata.json"
        write_json(synthesis.details["metadata"], metadata_path)
        paths["sdv_metadata"] = metadata_path
    dashboard_path = write_executive_dashboard(
        real_df,
        synthetic_df,
        mapping,
        validation,
        Path(args.out_dir) / "executive_dashboard.html",
        run_metadata=run_metadata,
    )
    paths["executive_dashboard"] = dashboard_path

    print(f"Synthetic rows: {synthetic_df.shape[0]}")
    if args.source_max_rows:
        print(f"Synthesis model limited to first {args.source_max_rows} source rows.")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")
    return 0


def _cmd_sensitive_profile(args: argparse.Namespace) -> int:
    df = read_csv(args.csv, nrows=args.max_rows)
    profile = field_profile(df)
    profile.update(
        {
            "source_file": Path(args.csv).name,
            "profile_row_limit": args.max_rows,
        }
    )
    target = Path(args.out)
    write_json(profile, target)
    print(f"Rows profiled: {df.shape[0]:,}")
    print(f"Fields profiled: {df.shape[1]:,}")
    print(f"Wrote sensitive field profile: {target}")
    return 0


def _cmd_synthesize_table(args: argparse.Namespace) -> int:
    header = read_csv(args.csv, nrows=0)
    selected = (
        [column.strip() for column in args.columns.split(",") if column.strip()]
        if args.columns
        else list(header.columns)
    )
    missing = [column for column in selected if column not in header.columns]
    if missing:
        raise ValueError(
            "Selected columns were not found: " + ", ".join(missing)
        )
    real_df = read_csv(args.csv, usecols=selected, nrows=args.source_max_rows)
    result = synthesize_single_table(
        real_df,
        selected_columns=selected,
        rows=args.rows,
        method=args.method,
        seed=args.seed,
        rare_threshold=args.rare_threshold,
        ctgan_epochs=args.ctgan_epochs,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "synthetic_data.csv"
    report_path = out_dir / "validation_report.json"
    metadata_path = out_dir / "sdv_metadata.json"
    dashboard_path = out_dir / "validation_dashboard.html"
    result.dataframe.to_csv(csv_path, index=False)
    write_json(result.report, report_path)
    write_json(result.report["metadata"], metadata_path)
    write_tabular_dashboard(
        result.model_data,
        result.dataframe,
        result.report,
        dashboard_path,
    )
    print(f"Synthetic rows: {len(result.dataframe):,}")
    print(f"SDV quality score: {result.report['quality']['overall_score']:.4f}")
    print(
        "Exact source identity matches: "
        f"{result.report['privacy']['direct_identifier_overlap']['exact_identity_combination']['matching_synthetic_rows']:,}"
    )
    print(f"Wrote dashboard: {dashboard_path}")
    print(f"Wrote synthetic CSV: {csv_path}")
    print(f"Wrote validation report: {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SyntheticCAD core pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("make-sample", help="Create a small CAD-like CSV")
    sample_parser.add_argument("--out", required=True, help="Output CSV path")
    sample_parser.add_argument("--seed", type=int, default=7)
    sample_parser.add_argument("--events", type=int, default=250)
    sample_parser.set_defaults(func=_cmd_make_sample)

    profile_parser = subparsers.add_parser("profile", help="Profile a CAD CSV and suggest mappings")
    profile_parser.add_argument("csv", help="Input CAD CSV")
    profile_parser.add_argument("--out-dir", default="outputs/profile")
    profile_parser.add_argument("--max-rows", type=int, default=None, help="Limit profiling to the first N rows")
    profile_parser.set_defaults(func=_cmd_profile)

    synth_parser = subparsers.add_parser("synthesize", help="Generate a synthetic CAD CSV")
    synth_parser.add_argument("csv", help="Input CAD CSV")
    synth_parser.add_argument("--mapping", required=True, help="Mapping JSON from profile step")
    synth_parser.add_argument("--out-dir", default="outputs/synthetic_run")
    synth_parser.add_argument("--events", type=int, default=None, help="Synthetic event count; default matches input")
    synth_parser.add_argument(
        "--method",
        choices=["conditional", "sdv"],
        default="conditional",
        help=(
            "CAD synthesis engine. Conditional is the implemented fast CAD model; "
            "SDV uses the event/unit relational path when unit-level data is present."
        ),
    )
    synth_parser.add_argument(
        "--source-max-rows",
        type=int,
        default=None,
        help="Limit synthesis modeling to the first N source rows for fast development runs",
    )
    synth_parser.add_argument("--seed", type=int, default=42)
    synth_parser.set_defaults(func=_cmd_synthesize)

    sensitive_parser = subparsers.add_parser(
        "sensitive-profile",
        help="Classify identifier, quasi-identifier, and sensitive fields",
    )
    sensitive_parser.add_argument("csv", help="Input CSV")
    sensitive_parser.add_argument(
        "--out",
        default="outputs/sensitive_profile.json",
    )
    sensitive_parser.add_argument("--max-rows", type=int, default=None)
    sensitive_parser.set_defaults(func=_cmd_sensitive_profile)

    table_parser = subparsers.add_parser(
        "synthesize-table",
        help="Generate a synthetic single-table dataset with SDV",
    )
    table_parser.add_argument("csv", help="Input CSV")
    table_parser.add_argument("--out-dir", default="outputs/tabular_run")
    table_parser.add_argument(
        "--columns",
        help="Comma-separated fields to include; default uses every field",
    )
    table_parser.add_argument("--rows", type=int, default=None)
    table_parser.add_argument(
        "--method",
        choices=["gaussian_copula", "ctgan"],
        default="gaussian_copula",
    )
    table_parser.add_argument("--rare-threshold", type=int, default=5)
    table_parser.add_argument("--ctgan-epochs", type=int, default=100)
    table_parser.add_argument("--source-max-rows", type=int, default=None)
    table_parser.add_argument("--seed", type=int, default=42)
    table_parser.set_defaults(func=_cmd_synthesize_table)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
