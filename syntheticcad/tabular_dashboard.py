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
    return f"""
      <div class="paired-metric neutral">
        <div class="metric-label">{escape(str(item.get("column", "")))}</div>
        <div class="paired-values">
          <span><b>{escape(_fmt(item.get("real_value")))}</b><small>{source_label}</small></span>
          <span><b>{escape(_fmt(item.get("synthetic_value")))}</b><small>{synthetic_label}</small></span>
        </div>
        <div class="metric-foot"><strong>{escape(str(item.get("metric", "Gap")))} {_fmt(item.get("gap"))}</strong><span>Lower means closer for this metric</span></div>
      </div>
    """


def _privacy_metric(label: str, value: Any, status: str, note: str) -> str:
    return f"""
      <div class="privacy-metric {status}">
        <div class="metric-label">{escape(label)}</div>
        <b>{escape(_fmt(value))}</b>
        <span>{escape(note)}</span>
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
        <div class="section-head"><div><h2>Record-distance Review Screen</h2><p>Sampled proximity evidence; no universal pass threshold is asserted.</p></div></div>
        <div class="privacy-grid">
          {_privacy_metric("Source reference to synthetic", synthetic_dcr, "neutral", "Median nearest-record distance in the sampled comparison.")}
          {_privacy_metric("Source-to-source benchmark", benchmark_dcr, "neutral", "Median distance between two sampled source subsets.")}
          {_privacy_metric("Median distance ratio", ratio, "neutral", "Synthetic proximity divided by the source benchmark.")}
          {_privacy_metric("Synthetic NNDR median", nndr_median, "neutral", "Nearest distance divided by second-nearest distance.")}
          {_privacy_metric("DCR lower-tail p01", source_to_synthetic.get("p01"), "neutral", "The closest 1% tail needs particular review.")}
          {_privacy_metric("NNDR lower-tail p01", nndr.get("p01"), "neutral", "Very low tail values may indicate unusually close rows.")}
        </div>
        <div class="evidence-notes">
          <p><strong>Comparison data:</strong> {escape(str(screens.get("comparison_data", "Source-reference subsets sampled from the same source data used to fit the synthesizer. This is not an independent holdout evaluation.")))}</p>
          <p><strong>Columns:</strong> {escape(", ".join(screens.get("columns", [])) or "None")} | <strong>Sample per group:</strong> {_fmt(screens.get("sample_size"))}</p>
          <p><strong>Distance and encoding:</strong> {escape(str(screens.get("distance", "Euclidean distance.")))} {escape(str(screens.get("encoding", "")))}</p>
          <p><strong>DCR ratio formula:</strong> {escape(str(distance.get("formula", "Not documented.")))}</p>
          <p><strong>NNDR formula:</strong> {escape(str(nndr_info.get("formula", "Not documented.")))}</p>
        </div>
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
          {_privacy_metric("Column distributions", properties.get("Column Shapes"), "neutral", f"SDV evaluated {len(shapes)} modeled fields.")}
          {_privacy_metric("Column-pair trends", pair_score, "neutral", pair_note)}
          {_privacy_metric("Basic validity gate", gate_label, "neutral", f"Diagnostic score {_fmt(diagnostic)}; checks validity, not privacy.")}
        </div>
        <div class="evidence-notes">
          <p><strong>Data used:</strong> {escape(str(scope.get("fidelity", "Full source data used for fitting compared with generated synthetic data; this is training-data fidelity.")))}</p>
          <p><strong>Overall score:</strong> {escape(str(scope.get("overall_score_meaning", "SDV overall quality is an aggregate of the quality components SDV could score in this run. See the component breakdown; it is not a privacy or row-level accuracy score.")))}</p>
          <p><strong>Diagnostic:</strong> {escape(str(scope.get("diagnostic_meaning", "A basic validity gate for data types, ranges, and structural rules; not a privacy score.")))}</p>
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
    windows = _sample_windows(real_df, synthetic_df, report)
    modeled_columns = pipeline.get("modeled_columns", [])
    excluded = pipeline.get("excluded_identifier_columns", [])
    methodology = report.get("methodology", "")
    method_label = pipeline.get("method", "").replace("_", " ").title()
    run_count = int(report.get("consistency", {}).get("run_count", 1))
    run_subtitle = "Shareable aggregate evidence; no real source records are embedded"
    total_label = "All runs total" if run_count > 1 else "Total"
    basic_runtime_label = "All runs time" if run_count > 1 else "Run time"
    distributions_json = json.dumps(distribution_payload).replace("</", "<\\/")
    samples_json = json.dumps(windows).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SyntheticCAD Validation</title>
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
    .offline {{ font-size:12px; font-weight:700; color:var(--review); }}
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
    .metric-foot {{ border-top:1px solid var(--line); margin-top:9px; padding-top:8px; display:flex; justify-content:space-between; gap:8px; font-size:10px; color:var(--muted); }}
    .privacy-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
    .privacy-metric {{ border:1px solid var(--line); padding:12px; }} .privacy-metric b {{ font-size:22px; display:block; margin:7px 0 4px; }}
    .privacy-metric span {{ color:var(--muted); font-size:11px; }}
    .privacy-metric.good {{ background:var(--good-bg); }} .privacy-metric.review {{ background:var(--review-bg); }} .privacy-metric.high {{ background:var(--high-bg); }}
    .chart-controls {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
    select,button.action {{ min-height:36px; border:1px solid #bcc6c0; border-radius:5px; background:#fff; padding:7px 10px; font:inherit; font-size:12px; }}
    #distribution-chart {{ height:280px; display:flex; align-items:flex-end; gap:4px; border-left:1px solid var(--line); border-bottom:1px solid var(--line); padding:18px 8px 28px; overflow:hidden; }}
    .bar-group {{ flex:1; min-width:10px; height:100%; display:flex; align-items:flex-end; justify-content:center; gap:2px; position:relative; }}
    .bar {{ width:42%; min-width:3px; }} .bar.real {{ background:var(--real); }} .bar.synthetic {{ background:var(--synthetic); }}
    .bar-group label {{ position:absolute; top:calc(100% + 7px); width:80px; left:50%; transform:translateX(-50%) rotate(-25deg); transform-origin:top center; font-size:9px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; text-align:center; }}
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
    .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .claims ul {{ margin:8px 0 0; padding-left:18px; font-size:12px; line-height:1.6; }}
    .good-text {{ color:var(--good); font-weight:750; }} .review-text {{ color:var(--review); font-weight:750; }} .high-text {{ color:var(--high); font-weight:750; }}
    .thresholds {{ font-size:11px; color:var(--muted); }}
    @media(max-width:950px) {{ .summary-strip,.paired-grid {{ grid-template-columns:repeat(2,1fr); }} .privacy-grid,.two-col,.sample-grid {{ grid-template-columns:1fr; }} }}
    @media(max-width:560px) {{ header {{ align-items:flex-start; }} .summary-strip,.paired-grid {{ grid-template-columns:1fr; }} .summary-item {{ border-right:0; border-bottom:1px solid var(--line); }} }}
  </style>
</head>
<body>
  <header>
    <div><h1>SyntheticCAD Validation</h1><div class="subtitle">{run_subtitle}</div></div>
    <div class="offline">Agency review required</div>
  </header>
  <nav>
    <button class="tab-button active" data-tab="basic">Basic Overview</button>
    <button class="tab-button" data-tab="advanced">Advanced Evidence</button>
  </nav>
  <main>
    <div id="basic" class="tab active">
      <div class="callout positioning"><strong>Evidence generated - agency review required.</strong> SyntheticCAD creates candidate synthetic data and an auditable evidence package for agency review. It does not certify that data is risk-free, legally unrestricted, or suitable for every research use.</div>
      <div class="summary-strip">
        <div class="summary-item"><small>Source rows evaluated</small><b>{_fmt(pipeline.get("source_rows"))}</b></div>
        <div class="summary-item"><small>Synthetic rows</small><b>{_fmt(pipeline.get("synthetic_rows"))}</b></div>
        <div class="summary-item"><small>Modeled fields</small><b>{_fmt(len(modeled_columns))}</b></div>
        <div class="summary-item"><small>Identifiers replaced</small><b>{_fmt(len(excluded))}</b></div>
        <div class="summary-item"><small>{basic_runtime_label}</small><b>{_fmt(runtime.get("total_all_runs_seconds", runtime.get("total_seconds")))} sec</b></div>
      </div>

      <section>
        <div class="section-head">
          <div><h2>Key Pattern Comparisons</h2><p>Four featured fields. The label on each tile names its statistic and gap formula; all fields are in Advanced Evidence.</p></div>
        </div>
        <div class="paired-grid">{paired or '<div class="callout">No comparable modeled fields were available.</div>'}</div>
      </section>

      <section>
        <div class="section-head">
          <div><h2>Privacy Evidence</h2><p>Observed overlap checks for this run. These are measurements, not a formal privacy guarantee.</p></div>
        </div>
        <div class="privacy-grid">
          {_privacy_metric("Exact identity combinations", direct.get("matching_synthetic_rows", 0), "neutral", f"{_fmt(direct.get('matching_synthetic_rows', 0))} of {_fmt(pipeline.get('synthetic_rows'))} synthetic rows ({_fmt(100 * direct.get('match_rate', 0), 2)}%). Fields: {', '.join(direct.get('columns', [])) or 'none selected'}.")}
          {_privacy_metric("Exact modeled rows", exact_rows.get("matching_synthetic_rows", 0), "neutral", f"{_fmt(exact_rows.get('matching_synthetic_rows', 0))} of {_fmt(pipeline.get('synthetic_rows'))} synthetic rows ({_fmt(100 * exact_rows.get('match_rate', 0), 2)}%) matched every modeled field.")}
          {_privacy_metric("Rare combinations present", rare.get("source_rare_combinations_present_in_synthetic", 0), "neutral", f"{_fmt(rare.get('source_rare_combinations_present_in_synthetic', 0))} of {_fmt(rare.get('source_rare_combinations', 0))} tested source combinations ({_fmt(100 * rare.get('presence_rate', 0), 1)}%).")}
        </div>
      </section>

      <section>
        <div class="section-head">
          <div><h2>Distribution Explorer</h2><p>Select a field to inspect the real and synthetic shape directly.</p></div>
          <div class="legend"><span><i class="dot" style="background:var(--real)"></i>Real</span><span><i class="dot" style="background:var(--synthetic)"></i>Synthetic</span></div>
        </div>
        <div class="chart-controls"><label for="field-select">Field</label><select id="field-select"></select></div>
        <div id="distribution-chart"></div>
      </section>

      <section>
        <div class="sample-head">
          <div><h2>Synthetic Record Spot Check</h2><div class="sample-meta" id="sample-label"></div></div>
          <button class="action" id="next-sample">Show another sample</button>
        </div>
        <div class="callout">This shareable dashboard contains no real source records. Source data can be inspected only inside the local application.</div>
        <div class="sample-meta" id="sample-counts"></div>
        <div class="sample-grid">
          <div class="sample-pane"><h3>Synthetic records</h3><div class="table-wrap" id="synthetic-sample"></div></div>
        </div>
      </section>
    </div>

    <div id="advanced" class="tab">
      <div class="summary-strip">
        <div class="summary-item"><small>Overall similarity (secondary)</small><b>{_fmt(quality.get("overall_score"))}</b></div>
        <div class="summary-item"><small>Basic validity gate</small><b>{"Passed" if quality.get("diagnostic_score") == 1.0 else "Review"}</b></div>
        <div class="summary-item"><small>Prepare</small><b>{_fmt(runtime.get("preparation_seconds"))} sec</b></div>
        <div class="summary-item"><small>Fit</small><b>{_fmt(runtime.get("fit_seconds"))} sec</b></div>
        <div class="summary-item"><small>Sample</small><b>{_fmt(runtime.get("sample_seconds"))} sec</b></div>
        <div class="summary-item"><small>Evaluate</small><b>{_fmt(runtime.get("evaluation_seconds"))} sec</b></div>
        <div class="summary-item"><small>{total_label}</small><b>{_fmt(runtime.get("total_all_runs_seconds", runtime.get("total_seconds")))} sec</b></div>
      </div>
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
        <div><h2>What This Run Supports</h2><ul>{"".join(f"<li>{escape(item)}</li>" for item in report.get("claims", {}).get("supported", []))}</ul></div>
        <div><h2>What This Run Does Not Claim</h2><ul>{"".join(f"<li>{escape(item)}</li>" for item in report.get("claims", {}).get("not_claimed", []))}</ul></div>
      </section>
      <section>
        <div class="section-head"><div><h2>Method and Trade-offs</h2><p>{escape(method_label)} | seed {_fmt(pipeline.get("seed"))} | rare category threshold {_fmt(pipeline.get("rare_category_threshold"))}</p></div></div>
        <div class="callout">{escape(methodology)}</div>
        <p class="thresholds">Differences occur because a synthesizer estimates distributions from finite data and then samples new rows. Rare-value grouping intentionally reduces fidelity for uncommon categories to lower disclosure risk. Neural methods may capture more complex relationships but require longer training and still do not create a formal privacy guarantee by themselves.</p>
      </section>
    </div>
  </main>
  <script>
    const distributions = {distributions_json};
    const samples = {samples_json};
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
