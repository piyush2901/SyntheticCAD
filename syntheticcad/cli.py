"""Command-line entry points for the SyntheticCAD core pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from syntheticcad.dashboard import write_executive_dashboard
from syntheticcad.mapping_guide import write_mapping_guide
from syntheticcad.profiling import build_profile, read_csv, write_json
from syntheticcad.sample_data import make_sample_cad
from syntheticcad.schema import SyntheticCADMapping, load_mapping, save_mapping
from syntheticcad.synthesis import synthesize_baseline, write_export_package
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
    synthetic_df = synthesize_baseline(
        real_df,
        mapping,
        event_count=args.events,
        seed=args.seed,
    )
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
        "synthetic_generation_method": (
            "Dependency-light baseline generator: sample mapped categorical "
            "distributions, preserve call-time patterns, create new event IDs, "
            "and replace mapped street-level location text with synthetic labels."
        ),
    }
    validation["generation_process"] = {
        "plain_language": (
            f"Read {real_df.shape[0]:,} real rows from {Path(args.csv).name}, "
            f"used the approved field mapping, and generated "
            f"{synthetic_df.shape[0]:,} synthetic rows locally."
        ),
        **run_metadata,
    }
    paths = write_export_package(synthetic_df, args.out_dir, validation_report=validation)
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

    synth_parser = subparsers.add_parser("synthesize", help="Generate a baseline synthetic CAD CSV")
    synth_parser.add_argument("csv", help="Input CAD CSV")
    synth_parser.add_argument("--mapping", required=True, help="Mapping JSON from profile step")
    synth_parser.add_argument("--out-dir", default="outputs/synthetic_run")
    synth_parser.add_argument("--events", type=int, default=None, help="Synthetic event count; default matches input")
    synth_parser.add_argument(
        "--source-max-rows",
        type=int,
        default=None,
        help="Limit synthesis modeling to the first N source rows for fast development runs",
    )
    synth_parser.add_argument("--seed", type=int, default=42)
    synth_parser.set_defaults(func=_cmd_synthesize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
