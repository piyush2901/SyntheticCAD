"""Interactive Basic/Advanced dashboard for general tabular synthesis runs."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from syntheticcad.dates import parse_datetime


REAL_COLOR = "#256f82"
SYNTHETIC_COLOR = "#d66a2c"


def write_tabular_dashboard(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    html = _render_dashboard(real_df, synthetic_df, report)
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    target.write_text(
        html,
        encoding="utf-8",
    )
    return target


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Not available"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _paired_metric(item: dict[str, Any]) -> str:
    sdtype = item.get("sdtype", "categorical")
    if sdtype == "numerical":
        source_label = "Source median"
        synthetic_label = "Synthetic median"
    elif sdtype == "datetime":
        source_label = "Source minimum"
        synthetic_label = "Synthetic minimum"
    else:
        source_label = "Source categories"
        synthetic_label = "Synthetic categories"
    gap = item.get("gap")
    if gap is None:
        gap_label = "Needs review"
        marker = 0
    else:
        gap_value = float(gap)
        marker = min(max(gap_value, 0.0), 1.0) * 100
        if gap_value <= 0.05:
            gap_label = "Very close"
        elif gap_value <= 0.10:
            gap_label = "Close"
        elif gap_value <= 0.20:
            gap_label = "Noticeable difference"
        else:
            gap_label = "Large difference"
    return f"""
      <div class="paired-metric neutral">
        <div class="metric-label">{escape(str(item.get("column", "")))}</div>
        <div class="paired-values">
          <span><b>{escape(_fmt(item.get("real_value")))}</b><small>{source_label}</small></span>
          <span><b>{escape(_fmt(item.get("synthetic_value")))}</b><small>{synthetic_label}</small></span>
        </div>
        <div class="gap-readout"><strong>{escape(str(item.get("metric", "Gap")))} {_fmt(gap)}</strong><span>{gap_label}</span></div>
        <div class="scale-track" title="0 means the distributions match; 1 means they are fully different"><i style="left:{marker:.1f}%"></i></div>
        <div class="scale-labels"><span>0 - same pattern</span><span>1 - different pattern</span></div>
      </div>
    """


def _privacy_metric(
    label: str,
    value: Any,
    status: str,
    note: str,
    guide: str = "",
) -> str:
    return f"""
      <div class="privacy-metric {status}">
        <div class="metric-label">{escape(label)}</div>
        <b>{escape(_fmt(value))}</b>
        <span>{escape(note)}</span>
        {f'<small class="how-read">How to read: {escape(guide)}</small>' if guide else ''}
      </div>
    """


def _distribution_payload(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    report: dict[str, Any],
) -> dict[str, Any]:
    assessments = {
        item["column"]: item for item in report.get("field_assessments", [])
    }
    payload: dict[str, Any] = {}
    for column in report.get("pipeline", {}).get("modeled_columns", []):
        if column not in real_df.columns or column not in synthetic_df.columns:
            continue
        sdtype = assessments.get(column, {}).get("sdtype", "categorical")
        real = real_df[column]
        synthetic = synthetic_df[column]
        if sdtype == "numerical":
            real_num = pd.to_numeric(real, errors="coerce").dropna()
            synth_num = pd.to_numeric(synthetic, errors="coerce").dropna()
            combined = pd.concat([real_num, synth_num])
            if combined.empty:
                continue
            minimum = float(combined.quantile(0.01))
            maximum = float(combined.quantile(0.99))
            if minimum == maximum:
                maximum = minimum + 1
            bins = np.linspace(minimum, maximum, 17)
            real_counts, _ = np.histogram(real_num.clip(minimum, maximum), bins=bins)
            synth_counts, _ = np.histogram(synth_num.clip(minimum, maximum), bins=bins)
            payload[column] = {
                "kind": "numeric",
                "labels": [f"{bins[index]:.1f}" for index in range(len(bins) - 1)],
                "real": (real_counts / max(real_counts.sum(), 1) * 100).round(3).tolist(),
                "synthetic": (
                    synth_counts / max(synth_counts.sum(), 1) * 100
                ).round(3).tolist(),
            }
        elif sdtype == "datetime":
            real_dt = parse_datetime(real).dropna()
            synth_dt = parse_datetime(synthetic).dropna()
            real_counts = real_dt.dt.to_period("M").astype(str).value_counts()
            synth_counts = synth_dt.dt.to_period("M").astype(str).value_counts()
            labels = sorted(real_counts.index.union(synth_counts.index))
            payload[column] = {
                "kind": "datetime",
                "labels": labels,
                "real": [
                    round(100 * real_counts.get(label, 0) / max(len(real_dt), 1), 3)
                    for label in labels
                ],
                "synthetic": [
                    round(100 * synth_counts.get(label, 0) / max(len(synth_dt), 1), 3)
                    for label in labels
                ],
            }
        else:
            real_counts = real.fillna("<Missing>").astype(str).value_counts()
            synth_counts = synthetic.fillna("<Missing>").astype(str).value_counts()
            rare_threshold = int(
                report.get("pipeline", {}).get("rare_category_threshold", 5)
            )
            common_source_counts = real_counts[real_counts >= rare_threshold]
            labels = list(
                (
                    common_source_counts / max(real_counts.sum(), 1)
                    + synth_counts.reindex(common_source_counts.index, fill_value=0)
                    / max(synth_counts.sum(), 1)
                )
                .sort_values(ascending=False)
                .head(12)
                .index
            )
            payload[column] = {
                "kind": "categorical",
                "labels": labels,
                "real": [
                    round(100 * real_counts.get(label, 0) / max(real_counts.sum(), 1), 3)
                    for label in labels
                ],
                "synthetic": [
                    round(
                        100 * synth_counts.get(label, 0) / max(synth_counts.sum(), 1),
                        3,
                    )
                    for label in labels
                ],
            }
    return payload


def _sample_windows(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    report: dict[str, Any],
    seed: int = 42,
) -> list[dict[str, Any]]:
    assessments = {
        item["column"]: item for item in report.get("field_assessments", [])
    }
    safe_columns = [
        column
        for column in report.get("pipeline", {}).get("modeled_columns", [])
        if column in real_df.columns and column in synthetic_df.columns
    ]
    date_columns = [
        column
        for column in safe_columns
        if assessments.get(column, {}).get("sdtype") == "datetime"
    ]
    display_columns = safe_columns[:8]
    rng = np.random.default_rng(seed)
    windows: list[dict[str, Any]] = []

    def records(frame: pd.DataFrame) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        for row in frame[display_columns].head(10).to_dict(orient="records"):
            output.append(
                {
                    column: "<Missing>" if pd.isna(value) else str(value)
                    for column, value in row.items()
                }
            )
        return output

    if date_columns and not parse_datetime(synthetic_df[date_columns[0]]).dropna().empty:
        column = date_columns[0]
        synthetic_dates = parse_datetime(synthetic_df[column])
        usable = synthetic_dates.dropna()
        has_time = bool(((usable.dt.hour != 0) | (usable.dt.minute != 0)).any())
        for _ in range(min(8, max(1, len(usable)))):
            anchor = usable.iloc[int(rng.integers(0, len(usable)))]
            if has_time:
                start = anchor.floor("5min")
                end = start + pd.Timedelta(minutes=5)
                label = f"{start} to {end} (5-minute window)"
            else:
                start = anchor.floor("D")
                end = start + pd.Timedelta(days=1)
                label = f"{start.date()} (source contains dates without times)"
            synth_mask = (synthetic_dates >= start) & (synthetic_dates < end)
            windows.append(
                {
                    "label": label,
                    "columns": display_columns,
                    "synthetic_count": int(synth_mask.sum()),
                    "synthetic_rows": records(synthetic_df.loc[synth_mask]),
                }
            )
    else:
        for _ in range(5):
            synthetic_sample = synthetic_df.sample(
                n=min(10, len(synthetic_df)),
                random_state=int(rng.integers(0, 1_000_000)),
            )
            windows.append(
                {
                    "label": "Random row sample (no modeled date/time field)",
                    "columns": display_columns,
                    "synthetic_count": len(synthetic_sample),
                    "synthetic_rows": records(synthetic_sample),
                }
            )
    return windows


def _advanced_rows(report: dict[str, Any]) -> str:
    rows: list[str] = []
    shape_scores = {
        item.get("Column"): item.get("Score")
        for item in report.get("quality", {}).get("column_shapes", [])
    }
    for item in report.get("quality", {}).get("column_metrics", []):
        gap = item.get("gap")
        score = shape_scores.get(item.get("column"))
        rows.append(
            f"""
            <tr>
              <td>{escape(str(item.get("column", "")))}</td>
              <td>{escape(str(item.get("sdtype", "")))}</td>
              <td data-sort="{item.get("real_value", "")}">{escape(_fmt(item.get("real_value")))}</td>
              <td data-sort="{item.get("synthetic_value", "")}">{escape(_fmt(item.get("synthetic_value")))}</td>
              <td data-sort="{gap if gap is not None else 999}">{escape(str(item.get("metric", "Gap")))}: {escape(_fmt(gap))}</td>
              <td data-sort="{score if score is not None else -1}">{escape(_fmt(score))}</td>
            </tr>
            """
        )
    return "".join(rows)


def _consistency_panel(report: dict[str, Any]) -> str:
    consistency = report.get("consistency", {})
    run_count = int(consistency.get("run_count", 1))
    if run_count <= 1:
        return """
        <section>
          <div class="section-head"><div><h2>Run Stability</h2><p>This dashboard contains one seed.</p></div></div>
          <div class="callout">One seed is exploratory evidence. Run three independent seeds before preparing a release-review package.</div>
        </section>
        """
    rows = "".join(
        f"""
        <tr>
          <td>{escape(str(item.get("seed", "")))}</td>
          <td>{escape(_fmt(item.get("quality_score"), 4))}</td>
          <td>{escape(_fmt(item.get("runtime_seconds")))} sec</td>
        </tr>
        """
        for item in consistency.get("runs", [])
    )
    spreads = consistency.get("field_gap_spread", {})
    largest_field, largest_spread = (
        max(spreads.items(), key=lambda item: item[1])
        if spreads
        else ("Not available", None)
    )
    return f"""
      <section>
        <div class="section-head"><div><h2>Run Stability</h2><p>{run_count} seeds were fitted and evaluated independently.</p></div></div>
        <div class="two-col">
          <div class="table-wrap"><table><thead><tr><th>Seed</th><th>SDV quality</th><th>Runtime</th></tr></thead><tbody>{rows}</tbody></table></div>
          <div class="callout">
            Quality range <strong>{_fmt(consistency.get("quality_score_min"), 4)} to {_fmt(consistency.get("quality_score_max"), 4)}</strong>.<br>
            Largest field-gap spread: <strong>{escape(str(largest_field))} / {_fmt(largest_spread, 4)}</strong>.
          </div>
        </div>
      </section>
    """


def _distance_panel(report: dict[str, Any]) -> str:
    screens = report.get("privacy", {}).get("distance_screens", {})
    if not screens.get("available"):
        return ""
    distance = screens.get("distance_to_closest_record", {})
    nndr_info = screens.get("nearest_neighbor_distance_ratio", {})
    nndr = nndr_info.get(
        "synthetic_to_source_benchmark",
        nndr_info.get("synthetic_to_real_train", {}),
    )
    source_to_synthetic = distance.get(
        "source_reference_to_synthetic",
        distance.get("holdout_to_synthetic", {}),
    )
    source_to_source = distance.get(
        "source_reference_to_source_benchmark",
        distance.get("holdout_to_real_train_benchmark", {}),
    )
    synthetic_dcr = source_to_synthetic.get("median")
    benchmark_dcr = source_to_source.get("median")
    ratio = distance.get("median_distance_ratio")
    nndr_median = nndr.get("median")
    return f"""
      <section>
        <div class="section-head"><div><h2>Record Closeness</h2><p>Check whether synthetic records sit unusually close to source records.</p></div></div>
        <div class="privacy-grid">
          {_privacy_metric("Typical source-to-synthetic distance", synthetic_dcr, "neutral", "Median nearest-record distance in the sample.", "Compare this with the source-to-source benchmark beside it.")}
          {_privacy_metric("Typical source-to-source distance", benchmark_dcr, "neutral", "Median distance between two source samples.", "This is the benchmark for interpreting synthetic distance.")}
          {_privacy_metric("Distance ratio", ratio, "neutral", "Synthetic distance divided by the source benchmark.", "Around 1 means similar spacing. Values well below 1 need closer review.")}
          {_privacy_metric("Nearest-neighbor ratio", nndr_median, "neutral", "Nearest distance divided by second-nearest distance.", "0 means one source row stands out as much closer; 1 means the two nearest are equally close.")}
          {_privacy_metric("Closest 1% distance", source_to_synthetic.get("p01"), "neutral", "Lower edge of the source-to-synthetic distances.", "Compare this tail with the typical distance; unusually small values need review.")}
          {_privacy_metric("Closest 1% neighbor ratio", nndr.get("p01"), "neutral", "Lower edge of nearest-neighbor ratios.", "Values near 0 indicate the most concerning tail of unusually close rows.")}
        </div>
        <div class="evidence-notes">
          <p><strong>Data used:</strong> {escape(str(screens.get("comparison_data", "Source-reference subsets sampled from the data used to fit the synthesizer. Run a separate holdout test before a production release.")))}</p>
          <p><strong>Columns:</strong> {escape(", ".join(screens.get("columns", [])) or "None")} | <strong>Sample per group:</strong> {_fmt(screens.get("sample_size"))}</p>
          <p><strong>Distance and encoding:</strong> {escape(str(screens.get("distance", "Euclidean distance.")))} {escape(str(screens.get("encoding", "")))}</p>
          <p><strong>DCR ratio formula:</strong> {escape(str(distance.get("formula", "Not documented.")))}</p>
          <p><strong>NNDR formula:</strong> {escape(str(nndr_info.get("formula", "Not documented.")))}</p>
        </div>
      </section>
    """


def _metric_guide_panel() -> str:
    return """
      <section>
        <div class="section-head"><div><h2>Metric Guide</h2><p>Use this scale before interpreting the numbers below.</p></div></div>
        <div class="guide-grid">
          <div><b>KS and TV distance</b><span>0 to 1. Lower is closer. 0 means the source and synthetic distributions match.</span></div>
          <div><b>SDV similarity</b><span>0 to 1. Higher is closer. 1 means the measured statistical pattern matches.</span></div>
          <div><b>Exact and rare matches</b><span>Counts and percentages. Lower is safer for these screens; review any repeated combinations.</span></div>
          <div><b>Distance ratio</b><span>Around 1 means similar record spacing. Values well below 1 need closer privacy review.</span></div>
          <div><b>Neighbor ratio</b><span>0 to 1. Values near 0 mean one source row is much closer than the second-nearest row.</span></div>
          <div><b>Validity gate</b><span>Passed means generated values follow the configured data types and ranges.</span></div>
        </div>
        <p class="thresholds">These explanations help navigate the evidence. Set release limits with the agency and researcher for the intended use.</p>
      </section>
    """


def _quality_composition_panel(report: dict[str, Any]) -> str:
    quality = report.get("quality", {})
    properties = {
        str(item.get("Property")): item.get("Score")
        for item in quality.get("properties", [])
    }
    shapes = quality.get("column_shapes", [])
    pairs = quality.get("column_pair_trends", [])
    modeled_count = len(report.get("pipeline", {}).get("modeled_columns", []))
    possible_pairs = modeled_count * (modeled_count - 1) // 2
    pair_score = properties.get("Column Pair Trends")
    pair_note = (
        f"{len(pairs)} pair records; aggregate score unavailable in this run."
        if pair_score is None
        else f"SDV scored {len(pairs)} of {possible_pairs} possible field pairs."
    )
    diagnostic = quality.get("diagnostic_score")
    gate_label = "Passed" if diagnostic is not None and float(diagnostic) == 1.0 else "Review"
    scope = quality.get("metric_scope", {})
    return f"""
      <section>
        <div class="section-head"><div><h2>Quality Composition</h2><p>Statistical resemblance, separated from privacy evidence.</p></div></div>
        <div class="privacy-grid">
          {_privacy_metric("Column distributions", properties.get("Column Shapes"), "neutral", f"SDV evaluated {len(shapes)} modeled fields.", "0 to 1; higher is closer, and 1 is the strongest measured match.")}
          {_privacy_metric("Column-pair trends", pair_score, "neutral", pair_note, "0 to 1; higher is closer. Unavailable means SDV did not return an aggregate score.")}
          {_privacy_metric("Basic validity gate", gate_label, "neutral", f"Diagnostic score {_fmt(diagnostic)}.", "Passed means values follow configured data types, ranges, and structural rules.")}
        </div>
        <div class="evidence-notes">
          <p><strong>Data used:</strong> {escape(str(scope.get("fidelity", "Full source data used for fitting compared with generated synthetic data; this is training-data fidelity.")))}</p>
          <p><strong>Overall score:</strong> {escape(str(scope.get("overall_score_meaning", "SDV combines the quality components it could score. Read the component scores above and use privacy checks separately.")))}</p>
          <p><strong>Validity:</strong> {escape(str(scope.get("diagnostic_meaning", "The validity gate checks data types, ranges, and structural rules. Use the privacy section for disclosure review.")))}</p>
        </div>
      </section>
    """


def _treatment_panel(report: dict[str, Any]) -> str:
    treatment = report.get("treatment_summary", {})
    pipeline = report.get("pipeline", {})
    rare_changes = treatment.get(
        "rare_category_changes",
        pipeline.get("rare_category_changes", {}),
    )
    grouped_fields = len(rare_changes)
    grouped_rows = sum(
        int(item.get("source_rows_grouped", 0)) for item in rare_changes.values()
    )
    repairs = treatment.get(
        "post_generation_repairs",
        pipeline.get("learned_constraints", []),
    )
    versions = report.get("software_versions", {})
    return f"""
      <section>
        <div class="section-head"><div><h2>Treatments, Repairs, and Versions</h2><p>What changed before fitting and after generation.</p></div></div>
        <div class="two-col">
          <div class="evidence-notes">
            <p><strong>Identifiers:</strong> {escape(str(treatment.get("identifier_treatment", "Direct and record identifiers were excluded from model fitting and regenerated after sampling.")))}</p>
            <p><strong>Excluded fields:</strong> {escape(", ".join(treatment.get("excluded_identifier_columns", pipeline.get("excluded_identifier_columns", []))) or "None")}</p>
            <p><strong>Rare-category grouping:</strong> threshold {_fmt(treatment.get("rare_category_threshold", pipeline.get("rare_category_threshold")))}; {grouped_fields} fields and {grouped_rows:,} source rows affected.</p>
          </div>
          <div class="evidence-notes">
            <p><strong>Type overrides:</strong> {len(treatment.get("metadata_type_overrides", pipeline.get("metadata_type_overrides", {})))}</p>
            <p><strong>Post-generation repairs:</strong> {len(repairs)} documented operations.</p>
            <p><strong>Software:</strong> {escape(" | ".join(f"{key} {value}" for key, value in versions.items()) or "Not documented")}</p>
          </div>
        </div>
      </section>
    """


def _render_dashboard(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    report: dict[str, Any],
) -> str:
    pipeline = report.get("pipeline", {})
    quality = report.get("quality", {})
    privacy = report.get("privacy", {})
    runtime = report.get("runtime", {})
    metrics = quality.get("column_metrics", [])
    overview_metrics = metrics[:4]
    paired = "".join(_paired_metric(item) for item in overview_metrics)
    direct = privacy.get("direct_identifier_overlap", {}).get(
        "exact_identity_combination", {}
    )
    exact_rows = privacy.get("exact_modeled_row_overlap", {})
    rare = privacy.get("rare_combination_exposure", {})
    distribution_payload = _distribution_payload(real_df, synthetic_df, report)
    trend_column = next(
        (
            column
            for column, payload in distribution_payload.items()
            if payload.get("kind") == "datetime"
        ),
        None,
    )
    trend_payload = (
        {"column": trend_column, **distribution_payload[trend_column]}
        if trend_column
        else {}
    )
    windows = _sample_windows(real_df, synthetic_df, report)
    modeled_columns = pipeline.get("modeled_columns", [])
    excluded = pipeline.get("excluded_identifier_columns", [])
    methodology = report.get("methodology", "")
    method_label = pipeline.get("method", "").replace("_", " ").title()
    run_count = int(report.get("consistency", {}).get("run_count", 1))
    run_subtitle = "Aggregate comparisons and synthetic samples only"
    total_label = "All runs total" if run_count > 1 else "Total"
    basic_runtime_label = "All runs time" if run_count > 1 else "Run time"
    distributions_json = json.dumps(distribution_payload).replace("</", "<\\/")
    trend_json = json.dumps(trend_payload).replace("</", "<\\/")
    samples_json = json.dumps(windows).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SyntheticCAD Evidence</title>
  <style>
    :root {{
      --ink:#17211f; --muted:#64706c; --line:#d7ddd9; --paper:#fff;
      --bg:#f3f5f2; --real:{REAL_COLOR}; --synthetic:{SYNTHETIC_COLOR};
      --good:#1d6b48; --good-bg:#e7f3ec; --review:#8a5b00; --review-bg:#fff2cf;
      --high:#a33b32; --high-bg:#f9e4e0; --accent:#245c55;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; overflow-x:hidden; font-family:"Segoe UI",Arial,sans-serif; letter-spacing:0; color:var(--ink); background:var(--bg); }}
    header {{ background:#fff; border-bottom:1px solid var(--line); padding:16px 24px; display:flex; align-items:center; justify-content:space-between; gap:16px; }}
    h1 {{ font-size:21px; margin:0; }}
    .subtitle {{ color:var(--muted); font-size:13px; margin-top:3px; }}
    .header-actions {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; justify-content:flex-end; }}
    .offline {{ font-size:12px; font-weight:700; color:var(--review); }}
    .hosted-link {{ min-height:34px; display:none; align-items:center; padding:7px 10px; border:1px solid var(--line); border-radius:5px; color:var(--ink); text-decoration:none; font-size:11px; font-weight:750; }}
    nav {{ max-width:1400px; margin:0 auto; padding:12px 20px 0; display:flex; gap:4px; }}
    nav button {{ border:1px solid var(--line); background:#fff; color:var(--ink); padding:9px 15px; border-radius:6px 6px 0 0; cursor:pointer; font-weight:700; }}
    nav button.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
    main {{ max-width:1400px; margin:0 auto; padding:16px 20px 36px; }}
    .tab {{ display:none; }} .tab.active {{ display:block; }}
    .summary-strip {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); border:1px solid var(--line); background:#fff; }}
    .summary-item {{ padding:13px 15px; border-right:1px solid var(--line); min-width:0; }}
    .summary-item:last-child {{ border-right:0; }}
    .summary-item small {{ color:var(--muted); display:block; font-size:11px; text-transform:uppercase; font-weight:700; }}
    .summary-item b {{ display:block; font-size:18px; margin-top:5px; overflow-wrap:anywhere; }}
    section {{ background:#fff; border:1px solid var(--line); margin-top:14px; padding:16px; }}
    .section-head {{ display:flex; justify-content:space-between; gap:14px; align-items:flex-start; margin-bottom:12px; }}
    h2 {{ font-size:15px; margin:0; }} .section-head p {{ margin:3px 0 0; color:var(--muted); font-size:12px; }}
    .legend {{ display:flex; gap:13px; font-size:12px; color:var(--muted); }}
    .dot {{ width:9px; height:9px; display:inline-block; margin-right:5px; border-radius:50%; }}
    .paired-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
    .paired-metric {{ border:1px solid var(--line); border-top:4px solid #8c9692; padding:11px; min-width:0; }}
    .metric-label {{ font-size:12px; font-weight:750; min-height:30px; }}
    .paired-values {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px; }}
    .paired-values span:first-child {{ color:var(--real); }} .paired-values span:last-child {{ color:var(--synthetic); }}
    .paired-values b {{ font-size:18px; display:block; }} .paired-values small {{ color:var(--muted); font-size:10px; }}
    .gap-readout {{ border-top:1px solid var(--line); margin-top:9px; padding-top:8px; display:flex; justify-content:space-between; gap:8px; font-size:10px; color:var(--muted); }}
    .gap-readout strong {{ color:var(--ink); }}
    .scale-track {{ height:7px; position:relative; margin-top:9px; background:linear-gradient(90deg,#31835f 0 10%,#e3b13f 10% 20%,#c65b4b 20% 100%); }}
    .scale-track i {{ position:absolute; top:-3px; width:2px; height:13px; background:#111; transform:translateX(-1px); }}
    .scale-labels {{ display:flex; justify-content:space-between; gap:8px; color:var(--muted); font-size:9px; margin-top:4px; }}
    .privacy-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
    .privacy-metric {{ border:1px solid var(--line); padding:12px; }} .privacy-metric b {{ font-size:22px; display:block; margin:7px 0 4px; }}
    .privacy-metric span {{ color:var(--muted); font-size:11px; display:block; line-height:1.45; }}
    .privacy-metric .how-read {{ display:block; border-top:1px solid var(--line); margin-top:9px; padding-top:8px; color:var(--ink); font-size:10px; line-height:1.45; }}
    .privacy-metric.good {{ background:var(--good-bg); }} .privacy-metric.review {{ background:var(--review-bg); }} .privacy-metric.high {{ background:var(--high-bg); }}
    .chart-controls {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
    select,button.action {{ min-height:36px; border:1px solid #bcc6c0; border-radius:5px; background:#fff; padding:7px 10px; font:inherit; font-size:12px; }}
    #distribution-chart {{ height:280px; display:flex; align-items:flex-end; gap:4px; border-left:1px solid var(--line); border-bottom:1px solid var(--line); padding:18px 8px 28px; overflow:hidden; }}
    .bar-group {{ flex:1; min-width:10px; height:100%; display:flex; align-items:flex-end; justify-content:center; gap:2px; position:relative; }}
    .bar {{ width:42%; min-width:3px; }} .bar.real {{ background:var(--real); }} .bar.synthetic {{ background:var(--synthetic); }}
    .bar-group label {{ position:absolute; top:calc(100% + 7px); width:80px; left:50%; transform:translateX(-50%) rotate(-25deg); transform-origin:top center; font-size:9px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; text-align:center; }}
    #trend-chart {{ width:100%; aspect-ratio:4/1; min-height:230px; border:1px solid var(--line); background:#fbfcfb; overflow:hidden; }}
    #trend-chart svg {{ width:100%; height:100%; display:block; }}
    .chart-note {{ color:var(--muted); font-size:11px; margin:8px 0 0; line-height:1.45; }}
    .sample-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; }}
    .sample-meta {{ color:var(--muted); font-size:12px; margin:8px 0; }}
    .sample-grid {{ display:grid; grid-template-columns:1fr; gap:12px; }}
    .sample-pane h3 {{ font-size:13px; margin:0 0 7px; color:var(--synthetic); }}
    .table-wrap {{ overflow:auto; max-height:320px; border:1px solid var(--line); }}
    table {{ border-collapse:collapse; width:100%; font-size:11px; background:#fff; }}
    th,td {{ border-bottom:1px solid #e4e8e5; padding:7px 8px; text-align:left; white-space:nowrap; }}
    th {{ position:sticky; top:0; background:#eef2ef; cursor:pointer; z-index:1; }}
    .callout {{ border-left:4px solid var(--accent); background:#eef5f3; padding:11px 13px; font-size:12px; line-height:1.5; }}
    .positioning {{ border-left-color:var(--review); background:var(--review-bg); margin-top:0; }}
    .evidence-notes {{ margin-top:12px; border:1px solid var(--line); padding:10px 12px; font-size:11px; color:var(--muted); line-height:1.5; }}
    .evidence-notes p {{ margin:4px 0; }}
    .guide-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); border:1px solid var(--line); }}
    .guide-grid div {{ padding:12px; border-right:1px solid var(--line); border-bottom:1px solid var(--line); min-width:0; }}
    .guide-grid div:nth-child(3n) {{ border-right:0; }} .guide-grid div:nth-last-child(-n+3) {{ border-bottom:0; }}
    .guide-grid b {{ display:block; font-size:11px; margin-bottom:5px; }} .guide-grid span {{ color:var(--muted); font-size:10px; line-height:1.45; }}
    .spot-details summary {{ cursor:pointer; font-size:13px; font-weight:750; }} .spot-details[open] summary {{ margin-bottom:12px; }}
    .run-files {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; }}
    .run-files a {{ border:1px solid var(--line); padding:11px; color:var(--ink); text-decoration:none; font-size:11px; font-weight:750; }}
    .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .claims ul {{ margin:8px 0 0; padding-left:18px; font-size:12px; line-height:1.6; }}
    .good-text {{ color:var(--good); font-weight:750; }} .review-text {{ color:var(--review); font-weight:750; }} .high-text {{ color:var(--high); font-weight:750; }}
    .thresholds {{ font-size:11px; color:var(--muted); }}
    @media(max-width:950px) {{ .summary-strip,.paired-grid {{ grid-template-columns:repeat(2,1fr); }} .privacy-grid,.two-col,.sample-grid,.guide-grid,.run-files {{ grid-template-columns:1fr; }} .guide-grid div {{ border-right:0; }} .guide-grid div:nth-last-child(-n+3) {{ border-bottom:1px solid var(--line); }} .guide-grid div:last-child {{ border-bottom:0; }} }}
    @media(max-width:560px) {{ header {{ align-items:flex-start; }} .summary-strip,.paired-grid {{ grid-template-columns:1fr; }} .summary-item {{ border-right:0; border-bottom:1px solid var(--line); }} }}
  </style>
</head>
<body>
  <header>
    <div><h1>SyntheticCAD Evidence</h1><div class="subtitle">{run_subtitle}</div></div>
    <div class="header-actions"><a class="hosted-link" id="hosted-demo-link" href="demo.html">Try the workflow</a><div class="offline">Review before sharing</div></div>
  </header>
  <nav>
    <button class="tab-button active" data-tab="basic">Basic Overview</button>
    <button class="tab-button" data-tab="advanced">Advanced Evidence</button>
  </nav>
  <main>
    <div id="basic" class="tab active">
      <div class="callout positioning"><strong>Start here.</strong> Check pattern differences and the privacy review section before sharing. Ask your privacy or legal reviewer to examine rare combinations and unusually close records.</div>
      <div class="summary-strip">
        <div class="summary-item"><small>Source rows evaluated</small><b>{_fmt(pipeline.get("source_rows"))}</b></div>
        <div class="summary-item"><small>Synthetic rows</small><b>{_fmt(pipeline.get("synthetic_rows"))}</b></div>
        <div class="summary-item"><small>Modeled fields</small><b>{_fmt(len(modeled_columns))}</b></div>
        <div class="summary-item"><small>Identifiers replaced</small><b>{_fmt(len(excluded))}</b></div>
        <div class="summary-item"><small>{basic_runtime_label}</small><b>{_fmt(runtime.get("total_all_runs_seconds", runtime.get("total_seconds")))} sec</b></div>
      </div>

      <section>
        <div class="section-head">
          <div><h2>How Well Key Patterns Match</h2><p>Each gap uses a 0-to-1 scale. Lower is closer; 0 means the measured distributions match.</p></div>
        </div>
        <div class="paired-grid">{paired or '<div class="callout">No comparable modeled fields were available.</div>'}</div>
        <p class="thresholds">The labels are a review guide: 0-0.05 very close, 0.05-0.10 close, 0.10-0.20 noticeable, and above 0.20 a large difference. Set final limits for the intended research use.</p>
      </section>

      <section id="trend-section">
        <div class="section-head">
          <div><h2>Pattern Over Time</h2><p id="trend-title">Source and synthetic event share by month.</p></div>
          <div class="legend"><span><i class="dot" style="background:var(--real)"></i>Source</span><span><i class="dot" style="background:var(--synthetic)"></i>Synthetic</span></div>
        </div>
        <div id="trend-chart"></div>
        <p class="chart-note">Lines that follow the same shape indicate that seasonality and changes over time were preserved. Hover over a point to see its percentage.</p>
      </section>

      <section>
        <div class="section-head">
          <div><h2>Privacy Review</h2><p>Use these screens to decide what needs closer inspection before release.</p></div>
        </div>
        <div class="privacy-grid">
          {_privacy_metric("Exact identity combinations", direct.get("matching_synthetic_rows", 0), "neutral", f"{_fmt(direct.get('matching_synthetic_rows', 0))} of {_fmt(pipeline.get('synthetic_rows'))} synthetic rows ({_fmt(100 * direct.get('match_rate', 0), 2)}%). Fields: {', '.join(direct.get('columns', [])) or 'none selected'}.", "Lower is better. Zero is the target for this exact-match screen.")}
          {_privacy_metric("Exact modeled rows", exact_rows.get("matching_synthetic_rows", 0), "neutral", f"{_fmt(exact_rows.get('matching_synthetic_rows', 0))} of {_fmt(pipeline.get('synthetic_rows'))} synthetic rows ({_fmt(100 * exact_rows.get('match_rate', 0), 2)}%) matched every modeled field.", "Lower is better. Review any exact rows and the fields that created them.")}
          {_privacy_metric("Rare combinations repeated", rare.get("source_rare_combinations_present_in_synthetic", 0), "neutral", f"{_fmt(rare.get('source_rare_combinations_present_in_synthetic', 0))} of {_fmt(rare.get('source_rare_combinations', 0))} tested source combinations ({_fmt(100 * rare.get('presence_rate', 0), 1)}%).", "Lower is better. Repeated rare combinations deserve focused linkage review.")}
        </div>
      </section>

      <section>
        <div class="section-head">
          <div><h2>Explore One Field</h2><p>Compare the full source and synthetic distribution for any modeled field.</p></div>
          <div class="legend"><span><i class="dot" style="background:var(--real)"></i>Real</span><span><i class="dot" style="background:var(--synthetic)"></i>Synthetic</span></div>
        </div>
        <div class="chart-controls"><label for="field-select">Field</label><select id="field-select"></select></div>
        <div id="distribution-chart"></div>
        <p class="chart-note" id="distribution-help">Each bar is the percentage of records in that value or range. Similar bar heights mean a closer match.</p>
      </section>

      <section>
        <details class="spot-details">
          <summary>Inspect synthetic sample records</summary>
          <div class="sample-head">
            <div><div class="sample-meta" id="sample-label"></div></div>
            <button class="action" id="next-sample">Show another sample</button>
          </div>
          <div class="callout">Only synthetic records appear here. Source records remain on the agency computer.</div>
          <div class="sample-meta" id="sample-counts"></div>
          <div class="sample-grid">
            <div class="sample-pane"><h3>Synthetic records</h3><div class="table-wrap" id="synthetic-sample"></div></div>
          </div>
        </details>
      </section>
    </div>

    <div id="advanced" class="tab">
      <div class="summary-strip">
        <div class="summary-item"><small>Overall similarity</small><b>{_fmt(quality.get("overall_score"))} / 1</b></div>
        <div class="summary-item"><small>Basic validity gate</small><b>{"Passed" if quality.get("diagnostic_score") == 1.0 else "Review"}</b></div>
        <div class="summary-item"><small>Prepare</small><b>{_fmt(runtime.get("preparation_seconds"))} sec</b></div>
        <div class="summary-item"><small>Fit</small><b>{_fmt(runtime.get("fit_seconds"))} sec</b></div>
        <div class="summary-item"><small>Sample</small><b>{_fmt(runtime.get("sample_seconds"))} sec</b></div>
        <div class="summary-item"><small>Evaluate</small><b>{_fmt(runtime.get("evaluation_seconds"))} sec</b></div>
        <div class="summary-item"><small>{total_label}</small><b>{_fmt(runtime.get("total_all_runs_seconds", runtime.get("total_seconds")))} sec</b></div>
      </div>
      {_metric_guide_panel()}
      {_quality_composition_panel(report)}
      <section>
        <div class="section-head"><div><h2>Sortable Field Metrics</h2><p>Click a column heading to sort. Source summary means median for numeric, minimum for datetime, and category count for categorical fields.</p></div></div>
        <div class="table-wrap" style="max-height:520px">
          <table id="metrics-table">
            <thead><tr><th>Field</th><th>Type</th><th>Source summary</th><th>Synthetic summary</th><th>Named gap</th><th>SDV shape score</th></tr></thead>
            <tbody>{_advanced_rows(report)}</tbody>
          </table>
        </div>
      </section>
      {_consistency_panel(report)}
      {_distance_panel(report)}
      {_treatment_panel(report)}
      <section class="two-col claims">
        <div><h2>Evidence Available Now</h2><ul>{"".join(f"<li>{escape(item)}</li>" for item in report.get("claims", {}).get("supported", []))}</ul></div>
        <div><h2>Before Sharing</h2><ul><li>Confirm quasi-identifiers with the agency privacy reviewer.</li><li>Set acceptance limits for the intended research questions.</li><li>Run an independent holdout evaluation for production use.</li><li>Review repeated rare combinations and unusually close records.</li></ul></div>
      </section>
      <section>
        <div class="section-head"><div><h2>Method and Trade-offs</h2><p>{escape(method_label)} | seed {_fmt(pipeline.get("seed"))} | rare category threshold {_fmt(pipeline.get("rare_category_threshold"))}</p></div></div>
        <div class="callout">{escape(methodology)}</div>
        <p class="thresholds">Some differences are expected because the model learns patterns and samples new records. Rare-value grouping deliberately changes uncommon categories. Compare another model only when an important research pattern needs improvement.</p>
      </section>
      <section class="hidden" id="run-files-section">
        <div class="section-head"><div><h2>Run Files</h2><p>Open a file, then use Back to evidence to return here.</p></div></div>
        <div class="run-files">
          <a id="run-synthetic" href="#">Download synthetic CSV</a>
          <a id="run-report" href="#">Validation report</a>
          <a id="run-metadata" href="#">SDV metadata</a>
          <a id="run-guidance" href="#">Sharing guidance</a>
        </div>
      </section>
    </div>
  </main>
  <script>
    const distributions = {distributions_json};
    const trend = {trend_json};
    const samples = {samples_json};
    if (location.hostname.endsWith('github.io')) {{
      document.getElementById('hosted-demo-link').style.display = 'inline-flex';
    }}
    document.querySelectorAll('.tab-button').forEach(button => button.addEventListener('click', () => {{
      document.querySelectorAll('.tab-button').forEach(item => item.classList.toggle('active', item === button));
      document.querySelectorAll('.tab').forEach(tab => tab.classList.toggle('active', tab.id === button.dataset.tab));
    }}));

    const select = document.getElementById('field-select');
    Object.keys(distributions).forEach(column => {{
      const option = document.createElement('option'); option.value = column; option.textContent = column; select.appendChild(option);
    }});
    function renderDistribution() {{
      const data = distributions[select.value]; const chart = document.getElementById('distribution-chart'); chart.innerHTML = '';
      if (!data) {{ chart.textContent = 'No distribution data available.'; return; }}
      const maximum = Math.max(1, ...data.real, ...data.synthetic);
      data.labels.forEach((label, index) => {{
        const group = document.createElement('div'); group.className = 'bar-group'; group.title = `${{label}} | Real ${{data.real[index]}}% | Synthetic ${{data.synthetic[index]}}%`;
        const real = document.createElement('div'); real.className = 'bar real'; real.style.height = `${{100 * data.real[index] / maximum}}%`;
        const synthetic = document.createElement('div'); synthetic.className = 'bar synthetic'; synthetic.style.height = `${{100 * data.synthetic[index] / maximum}}%`;
        const text = document.createElement('label'); text.textContent = label;
        group.append(real, synthetic, text); chart.appendChild(group);
      }});
    }}
    select.addEventListener('change', renderDistribution); renderDistribution();

    function svgElement(name, attributes={{}}) {{
      const element = document.createElementNS('http://www.w3.org/2000/svg', name);
      Object.entries(attributes).forEach(([key,value]) => element.setAttribute(key, String(value)));
      return element;
    }}
    function smoothPath(points) {{
      if (!points.length) return '';
      if (points.length === 1) return `M ${{points[0].x}} ${{points[0].y}}`;
      let path = `M ${{points[0].x}} ${{points[0].y}}`;
      for (let index=1; index<points.length; index += 1) {{
        const previous=points[index-1], current=points[index], midpoint=(current.x-previous.x)/2;
        path += ` C ${{previous.x+midpoint}} ${{previous.y}}, ${{current.x-midpoint}} ${{current.y}}, ${{current.x}} ${{current.y}}`;
      }}
      return path;
    }}
    function renderTrend() {{
      const section=document.getElementById('trend-section');
      if (!trend.labels || !trend.labels.length) {{ section.classList.add('hidden'); return; }}
      document.getElementById('trend-title').textContent = `${{trend.column}}: source and synthetic share by month.`;
      const container=document.getElementById('trend-chart');
      const width=1000, height=260, left=52, right=22, top=22, bottom=42;
      const maximum=Math.max(1,...trend.real,...trend.synthetic);
      const x=index => left + index*(width-left-right)/Math.max(trend.labels.length-1,1);
      const y=value => top + (maximum-value)*(height-top-bottom)/maximum;
      const svg=svgElement('svg',{{viewBox:`0 0 ${{width}} ${{height}}`,role:'img','aria-label':`${{trend.column}} trend comparison`}});
      [0,.25,.5,.75,1].forEach(fraction => {{
        const line=svgElement('line',{{x1:left,x2:width-right,y1:top+fraction*(height-top-bottom),y2:top+fraction*(height-top-bottom),stroke:'#dfe5e1','stroke-width':1}}); svg.appendChild(line);
      }});
      const sourcePoints=trend.real.map((value,index)=>({{x:x(index),y:y(value),value}}));
      const syntheticPoints=trend.synthetic.map((value,index)=>({{x:x(index),y:y(value),value}}));
      svg.appendChild(svgElement('path',{{d:smoothPath(sourcePoints),fill:'none',stroke:'{REAL_COLOR}','stroke-width':4,'stroke-linecap':'round'}}));
      svg.appendChild(svgElement('path',{{d:smoothPath(syntheticPoints),fill:'none',stroke:'{SYNTHETIC_COLOR}','stroke-width':4,'stroke-linecap':'round'}}));
      const labelEvery=Math.max(1,Math.ceil(trend.labels.length/6));
      trend.labels.forEach((label,index) => {{
        if (index % labelEvery === 0 || index === trend.labels.length-1) {{
          const text=svgElement('text',{{x:x(index),y:height-14,'text-anchor':'middle',fill:'#64706c','font-size':11}}); text.textContent=label; svg.appendChild(text);
        }}
        [[sourcePoints[index],'{REAL_COLOR}','Source'],[syntheticPoints[index],'{SYNTHETIC_COLOR}','Synthetic']].forEach(([point,color,name]) => {{
          const circle=svgElement('circle',{{cx:point.x,cy:point.y,r:4,fill:color,stroke:'#fff','stroke-width':2}});
          const title=svgElement('title'); title.textContent=`${{label}} | ${{name}} ${{point.value}}%`; circle.appendChild(title); svg.appendChild(circle);
        }});
      }});
      const yLabel=svgElement('text',{{x:12,y:16,fill:'#64706c','font-size':11}}); yLabel.textContent=`Share of records (max ${{maximum.toFixed(1)}}%)`; svg.appendChild(yLabel);
      container.replaceChildren(svg);
    }}
    renderTrend();

    let sampleIndex = 0;
    function sampleTable(columns, rows) {{
      if (!rows.length) return '<div class="sample-meta" style="padding:10px">No records in this window.</div>';
      return `<table><thead><tr>${{columns.map(column => `<th>${{escapeHtml(column)}}</th>`).join('')}}</tr></thead><tbody>${{rows.map(row => `<tr>${{columns.map(column => `<td>${{escapeHtml(row[column] ?? '')}}</td>`).join('')}}</tr>`).join('')}}</tbody></table>`;
    }}
    function escapeHtml(value) {{ const div = document.createElement('div'); div.textContent = String(value); return div.innerHTML; }}
    function renderSample() {{
      if (!samples.length) return;
      const item = samples[sampleIndex % samples.length];
      document.getElementById('sample-label').textContent = item.label;
      document.getElementById('sample-counts').textContent = `Synthetic records in this view: ${{item.synthetic_count.toLocaleString()}}`;
      document.getElementById('synthetic-sample').innerHTML = sampleTable(item.columns, item.synthetic_rows);
    }}
    document.getElementById('next-sample').addEventListener('click', () => {{ sampleIndex += 1; renderSample(); }}); renderSample();

    const artifactPrefix='/artifacts/';
    if (location.pathname.startsWith(artifactPrefix)) {{
      const dashboardRelative=decodeURIComponent(location.pathname.slice(artifactPrefix.length));
      const separator=Math.max(dashboardRelative.lastIndexOf('/'),dashboardRelative.lastIndexOf('\\\\'));
      const directory=separator >= 0 ? dashboardRelative.slice(0,separator+1) : '';
      const viewer=name => `/artifact-view?path=${{encodeURIComponent(directory+name)}}&back=${{encodeURIComponent(dashboardRelative)}}`;
      document.getElementById('run-synthetic').href=`/artifacts/${{directory.split('\\\\').join('/')}}synthetic_data.csv?download=1`;
      document.getElementById('run-synthetic').setAttribute('download','');
      document.getElementById('run-report').href=viewer('validation_report.json');
      document.getElementById('run-metadata').href=viewer('sdv_metadata.json');
      document.getElementById('run-guidance').href=viewer('disclaimer.txt');
      document.querySelectorAll('#run-files-section a:not([download])').forEach(link => link.target='_blank');
      document.getElementById('run-files-section').classList.remove('hidden');
    }}

    document.querySelectorAll('#metrics-table th').forEach((header, index) => header.addEventListener('click', () => {{
      const body = document.querySelector('#metrics-table tbody'); const rows = [...body.querySelectorAll('tr')];
      const ascending = header.dataset.direction !== 'asc'; header.dataset.direction = ascending ? 'asc' : 'desc';
      rows.sort((left, right) => {{
        const a = left.children[index].dataset.sort ?? left.children[index].textContent;
        const b = right.children[index].dataset.sort ?? right.children[index].textContent;
        const an = Number(a), bn = Number(b); const result = Number.isNaN(an) || Number.isNaN(bn) ? a.localeCompare(b) : an - bn;
        return ascending ? result : -result;
      }});
      rows.forEach(row => body.appendChild(row));
    }}));
  </script>
</body>
</html>"""
