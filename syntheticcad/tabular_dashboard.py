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
    target.write_text(
        _render_dashboard(real_df, synthetic_df, report),
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


def _status_class(gap: Any) -> str:
    if gap is None:
        return "neutral"
    value = float(gap)
    if value <= 0.1:
        return "good"
    if value < 0.5:
        return "review"
    return "high"


def _paired_metric(label: str, real: Any, synthetic: Any, gap: Any, note: str) -> str:
    return f"""
      <div class="paired-metric {_status_class(gap)}">
        <div class="metric-label">{escape(label)}</div>
        <div class="paired-values">
          <span><b>{escape(_fmt(real))}</b><small>Real</small></span>
          <span><b>{escape(_fmt(synthetic))}</b><small>Synthetic</small></span>
        </div>
        <div class="metric-foot"><strong>Gap {_fmt(gap)}</strong><span>{escape(note)}</span></div>
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
            labels = list(
                (
                    real_counts / max(real_counts.sum(), 1)
                    + synth_counts / max(synth_counts.sum(), 1)
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

    if date_columns:
        column = date_columns[0]
        real_dates = parse_datetime(real_df[column])
        synthetic_dates = parse_datetime(synthetic_df[column])
        usable = real_dates.dropna()
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
            real_mask = (real_dates >= start) & (real_dates < end)
            synth_mask = (synthetic_dates >= start) & (synthetic_dates < end)
            windows.append(
                {
                    "label": label,
                    "columns": display_columns,
                    "real_count": int(real_mask.sum()),
                    "synthetic_count": int(synth_mask.sum()),
                    "real_rows": records(real_df.loc[real_mask]),
                    "synthetic_rows": records(synthetic_df.loc[synth_mask]),
                }
            )
    else:
        for _ in range(5):
            real_sample = real_df.sample(
                n=min(10, len(real_df)),
                random_state=int(rng.integers(0, 1_000_000)),
            )
            synthetic_sample = synthetic_df.sample(
                n=min(10, len(synthetic_df)),
                random_state=int(rng.integers(0, 1_000_000)),
            )
            windows.append(
                {
                    "label": "Random row sample (no modeled date/time field)",
                    "columns": display_columns,
                    "real_count": len(real_sample),
                    "synthetic_count": len(synthetic_sample),
                    "real_rows": records(real_sample),
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
              <td data-sort="{gap if gap is not None else 999}" class="{_status_class(gap)}-text">{escape(_fmt(gap))}</td>
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
          <div class="callout">Choose the three-seed stability check in the local app to measure whether quality and field gaps remain consistent across repeated runs.</div>
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
    nndr = screens.get("nearest_neighbor_distance_ratio", {}).get(
        "synthetic_to_real_train",
        {},
    )
    synthetic_dcr = distance.get("holdout_to_synthetic", {}).get("median")
    benchmark_dcr = distance.get("holdout_to_real_train_benchmark", {}).get(
        "median"
    )
    ratio = distance.get("median_distance_ratio")
    ratio_status = (
        "good" if ratio is not None and ratio >= 0.8 else "review"
    )
    nndr_median = nndr.get("median")
    nndr_status = (
        "good" if nndr_median is not None and nndr_median >= 0.5 else "review"
    )
    return f"""
      <section>
        <div class="section-head"><div><h2>Record Distance Screens</h2><p>Sampled proximity checks benchmarked against a protected real holdout.</p></div></div>
        <div class="privacy-grid">
          {_privacy_metric("Holdout to synthetic median", synthetic_dcr, "neutral", "Distance to the closest sampled synthetic row.")}
          {_privacy_metric("Holdout to real benchmark", benchmark_dcr, "neutral", "Distance between held-out and training real rows.")}
          {_privacy_metric("DCR benchmark ratio", ratio, ratio_status, "Values materially below 1 require review.")}
          {_privacy_metric("Synthetic NNDR median", nndr_median, nndr_status, "Values near zero indicate a row may be unusually close to one training row.")}
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
    paired = "".join(
        _paired_metric(
            item["column"],
            item.get("real_value"),
            item.get("synthetic_value"),
            item.get("gap"),
            item.get("metric", ""),
        )
        for item in overview_metrics
    )
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
    run_subtitle = (
        f"{run_count} independent local synthesis runs, shown without source identifiers"
        if run_count > 1
        else "One local synthesis run, shown without source identifiers"
    )
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
    .offline {{ font-size:12px; font-weight:700; color:var(--good); }}
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
    .paired-metric.good {{ border-top-color:var(--good); }} .paired-metric.review {{ border-top-color:#d29a21; }} .paired-metric.high {{ border-top-color:var(--high); }}
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
    .sample-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    .sample-pane h3 {{ font-size:13px; margin:0 0 7px; }} .sample-pane:first-child h3 {{ color:var(--real); }} .sample-pane:last-child h3 {{ color:var(--synthetic); }}
    .table-wrap {{ overflow:auto; max-height:320px; border:1px solid var(--line); }}
    table {{ border-collapse:collapse; width:100%; font-size:11px; background:#fff; }}
    th,td {{ border-bottom:1px solid #e4e8e5; padding:7px 8px; text-align:left; white-space:nowrap; }}
    th {{ position:sticky; top:0; background:#eef2ef; cursor:pointer; z-index:1; }}
    .callout {{ border-left:4px solid var(--accent); background:#eef5f3; padding:11px 13px; font-size:12px; line-height:1.5; }}
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
    <div class="offline">Processed locally</div>
  </header>
  <nav>
    <button class="tab-button active" data-tab="basic">Basic Overview</button>
    <button class="tab-button" data-tab="advanced">Advanced Evidence</button>
  </nav>
  <main>
    <div id="basic" class="tab active">
      <div class="summary-strip">
        <div class="summary-item"><small>Real rows</small><b>{_fmt(pipeline.get("source_rows"))}</b></div>
        <div class="summary-item"><small>Synthetic rows</small><b>{_fmt(pipeline.get("synthetic_rows"))}</b></div>
        <div class="summary-item"><small>Modeled fields</small><b>{_fmt(len(modeled_columns))}</b></div>
        <div class="summary-item"><small>Identifiers replaced</small><b>{_fmt(len(excluded))}</b></div>
        <div class="summary-item"><small>{basic_runtime_label}</small><b>{_fmt(runtime.get("total_all_runs_seconds", runtime.get("total_seconds")))} sec</b></div>
      </div>

      <section>
        <div class="section-head">
          <div><h2>Key Pattern Comparisons</h2><p>Each tile uses one real value, one synthetic value, and the measured distribution gap.</p></div>
          <div class="thresholds">Gap <= 0.10 green | 0.10-0.50 review | >= 0.50 high</div>
        </div>
        <div class="paired-grid">{paired or '<div class="callout">No comparable modeled fields were available.</div>'}</div>
      </section>

      <section>
        <div class="section-head">
          <div><h2>Privacy Evidence</h2><p>Observed overlap checks for this run. These are measurements, not a formal privacy guarantee.</p></div>
        </div>
        <div class="privacy-grid">
          {_privacy_metric("Exact source identities reproduced", direct.get("matching_synthetic_rows", 0), "good" if direct.get("matching_synthetic_rows", 0) == 0 else "high", "Uses all selected direct identifier fields together.")}
          {_privacy_metric("Exact modeled rows reproduced", exact_rows.get("matching_synthetic_rows", 0), "good" if exact_rows.get("matching_synthetic_rows", 0) == 0 else "review", "Exact agreement across every modeled field.")}
          {_privacy_metric("Rare source combinations exposed", rare.get("source_rare_combinations_present_in_synthetic", 0), "good" if rare.get("presence_rate", 0) <= 0.1 else "review", f"{_fmt(100 * rare.get('presence_rate', 0), 1)}% of tested rare combinations.")}
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
          <div><h2>Random Sample Check</h2><div class="sample-meta" id="sample-label"></div></div>
          <button class="action" id="next-sample">Show another sample</button>
        </div>
        <div class="sample-meta" id="sample-counts"></div>
        <div class="sample-grid">
          <div class="sample-pane"><h3>Real records (identifier fields excluded)</h3><div class="table-wrap" id="real-sample"></div></div>
          <div class="sample-pane"><h3>Synthetic records</h3><div class="table-wrap" id="synthetic-sample"></div></div>
        </div>
      </section>
    </div>

    <div id="advanced" class="tab">
      <div class="summary-strip">
        <div class="summary-item"><small>SDV overall quality</small><b>{_fmt(quality.get("overall_score"))}</b></div>
        <div class="summary-item"><small>SDV diagnostic</small><b>{_fmt(quality.get("diagnostic_score"))}</b></div>
        <div class="summary-item"><small>Prepare</small><b>{_fmt(runtime.get("preparation_seconds"))} sec</b></div>
        <div class="summary-item"><small>Fit</small><b>{_fmt(runtime.get("fit_seconds"))} sec</b></div>
        <div class="summary-item"><small>Sample</small><b>{_fmt(runtime.get("sample_seconds"))} sec</b></div>
        <div class="summary-item"><small>Evaluate</small><b>{_fmt(runtime.get("evaluation_seconds"))} sec</b></div>
        <div class="summary-item"><small>{total_label}</small><b>{_fmt(runtime.get("total_all_runs_seconds", runtime.get("total_seconds")))} sec</b></div>
      </div>
      <section>
        <div class="section-head"><div><h2>Sortable Field Metrics</h2><p>Click a column heading to sort. Lower gap is better; higher SDV shape score is better.</p></div></div>
        <div class="table-wrap" style="max-height:520px">
          <table id="metrics-table">
            <thead><tr><th>Field</th><th>Type</th><th>Real</th><th>Synthetic</th><th>Gap</th><th>SDV shape score</th></tr></thead>
            <tbody>{_advanced_rows(report)}</tbody>
          </table>
        </div>
      </section>
      {_consistency_panel(report)}
      {_distance_panel(report)}
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
      document.getElementById('sample-counts').textContent = `Real ${{item.real_count.toLocaleString()}} | Synthetic ${{item.synthetic_count.toLocaleString()}}`;
      document.getElementById('real-sample').innerHTML = sampleTable(item.columns, item.real_rows);
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
