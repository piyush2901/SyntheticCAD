"""Executive dashboard generation for SyntheticCAD."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from syntheticcad.dates import parse_datetime
from syntheticcad.schema import SyntheticCADMapping


DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def write_executive_dashboard(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    validation_report: dict[str, Any],
    out_path: str | Path,
    run_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a standalone, non-technical HTML validation dashboard."""

    metadata = run_metadata or {}
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    html = _render_dashboard(real_df, synthetic_df, mapping, validation_report, metadata)
    target.write_text(html, encoding="utf-8")
    return target


def _render_dashboard(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    validation_report: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    summary = validation_report.get("summary", {})
    real_rows = int(summary.get("real_rows", real_df.shape[0]))
    synthetic_rows = int(summary.get("synthetic_rows", synthetic_df.shape[0]))
    real_columns = int(summary.get("real_columns", real_df.shape[1]))
    synthetic_columns = int(summary.get("synthetic_columns", synthetic_df.shape[1]))
    source_rows = metadata.get("source_rows_used", real_rows)
    source_limit = metadata.get("source_row_limit")
    source_columns_found = metadata.get("source_columns_found")
    mapped_columns_used = metadata.get("mapped_columns_used", real_columns)
    requested_events = metadata.get("requested_synthetic_events")

    call_volume_gap = validation_report.get("call_volume", {}).get(
        "day_of_week_hour_mean_abs_pct_point_gap"
    )
    call_type_gap = _largest_gap(validation_report, "call_type")
    priority_gap = _largest_gap(validation_report, "priority")

    plain_summary = _plain_language_summary(
        real_rows=real_rows,
        synthetic_rows=synthetic_rows,
        call_volume_gap=call_volume_gap,
        call_type_gap=call_type_gap,
        priority_gap=priority_gap,
        mapping=mapping,
    )

    limitations = _limitations(validation_report, mapping)
    geography_column = _select_geography_column(real_df, synthetic_df, mapping)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SyntheticCAD Executive Validation Dashboard</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #5d6875;
      --line: #d7dde5;
      --soft: #f5f7fa;
      --panel: #ffffff;
      --blue: #2563eb;
      --green: #0f766e;
      --amber: #b45309;
      --red: #b91c1c;
      --real: #2563eb;
      --synthetic: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: #eef2f6;
      line-height: 1.45;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px;
    }}
    header {{
      padding: 28px 0 20px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 19px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 15px;
      letter-spacing: 0;
    }}
    p {{ margin: 0 0 10px; }}
    .subtitle {{
      color: var(--muted);
      max-width: 920px;
      font-size: 16px;
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      margin: 16px 0;
    }}
    .grid {{
      display: grid;
      gap: 14px;
    }}
    .metrics {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    .two {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--soft);
      min-height: 112px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric .value {{
      font-size: 26px;
      font-weight: 700;
      margin-bottom: 6px;
      overflow-wrap: anywhere;
    }}
    .metric .note {{
      color: var(--muted);
      font-size: 13px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      font-weight: 700;
      color: #fff;
      background: var(--green);
      margin-right: 6px;
      margin-bottom: 6px;
    }}
    .badge.warn {{ background: var(--amber); }}
    .badge.stop {{ background: var(--red); }}
    .steps {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .step {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfe;
    }}
    .step-number {{
      width: 26px;
      height: 26px;
      border-radius: 50%;
      background: var(--blue);
      color: #fff;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .chart-panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: #fff;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }}
    .swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      margin-right: 6px;
      vertical-align: -1px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(120px, 220px) minmax(0, 1fr) 72px;
      gap: 12px;
      align-items: center;
      margin: 12px 0;
      font-size: 13px;
    }}
    .bar-label {{
      overflow-wrap: anywhere;
      color: var(--ink);
      font-weight: 600;
    }}
    .compare-bars {{
      display: grid;
      gap: 5px;
      min-width: 0;
    }}
    .mini-row {{
      display: grid;
      grid-template-columns: 70px minmax(0, 1fr) 48px;
      gap: 8px;
      align-items: center;
    }}
    .series-name {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .series-name.real {{ color: var(--real); }}
    .series-name.synthetic {{ color: var(--synthetic); }}
    .compare-track {{
      height: 12px;
      background: #edf1f5;
      border-radius: 4px;
      overflow: hidden;
    }}
    .compare-fill {{
      height: 100%;
      border-radius: 4px;
    }}
    .compare-fill.real {{ background: var(--real); opacity: .86; }}
    .compare-fill.synthetic {{ background: var(--synthetic); opacity: .86; }}
    .mini-value {{
      color: var(--muted);
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-size: 12px;
    }}
    .bar-value {{
      color: var(--muted);
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-size: 12px;
    }}
    .gap-value {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 26px;
      border-radius: 999px;
      background: #f1f5f9;
      color: #42505f;
      font-size: 12px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      padding: 4px 7px;
    }}
    .heatmaps {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .heatmap-title {{
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .heatmap {{
      display: grid;
      grid-template-columns: 34px repeat(24, minmax(10px, 1fr));
      gap: 2px;
      align-items: center;
    }}
    .geo-grid {{
      display: grid;
      grid-template-columns: repeat(12, minmax(12px, 1fr));
      gap: 3px;
      align-items: center;
    }}
    .geo-cell {{
      aspect-ratio: 1 / 1;
      min-height: 13px;
      border-radius: 3px;
      background: #eef2f6;
    }}
    .curve-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: #fff;
      margin-top: 14px;
    }}
    .curve-svg {{
      width: 100%;
      height: auto;
      display: block;
      margin-top: 8px;
    }}
    .stat-note {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
    }}
    .heatmap .hour, .heatmap .day {{
      color: var(--muted);
      font-size: 10px;
      text-align: center;
    }}
    .heatmap .day {{ text-align: right; padding-right: 4px; }}
    .cell {{
      aspect-ratio: 1 / 1;
      min-height: 10px;
      border-radius: 2px;
      background: #eef2f6;
    }}
    .callout {{
      border-left: 4px solid var(--blue);
      background: #f7faff;
      padding: 12px 14px;
      border-radius: 6px;
      color: var(--ink);
    }}
    .limitations {{
      margin: 0;
      padding-left: 20px;
    }}
    .limitations li {{ margin: 7px 0; }}
    footer {{
      color: var(--muted);
      font-size: 12px;
      padding: 18px 0 8px;
    }}
    @media (max-width: 900px) {{
      main {{ padding: 18px; }}
      .metrics, .two, .steps, .heatmaps {{
        grid-template-columns: 1fr;
      }}
      .bar-row {{
        grid-template-columns: 1fr;
      }}
      .mini-row {{
        grid-template-columns: 62px minmax(0, 1fr) 44px;
      }}
      .heatmap {{
        grid-template-columns: 30px repeat(24, minmax(7px, 1fr));
      }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>SyntheticCAD Executive Validation Dashboard</h1>
    <p class="subtitle">{escape(plain_summary)}</p>
  </header>

  <section class="section">
    <h2>Run Summary</h2>
    <div class="grid metrics">
      {_metric_card("Real rows used", _format_int(real_rows), _source_note(source_limit, source_rows))}
      {_metric_card("Synthetic rows created", _format_int(synthetic_rows), _requested_note(requested_events))}
      {_metric_card("Mapped fields used", _field_value(mapped_columns_used, source_columns_found, synthetic_columns), _field_note(source_columns_found))}
      {_metric_card("Call pattern match", _quality_label(call_volume_gap), _gap_note(call_volume_gap))}
    </div>
  </section>

  <section class="section">
    <h2>What The Tool Did</h2>
    <div class="callout">
      SyntheticCAD read the source file locally, learned high-level patterns from the mapped fields, generated new records with new event identifiers, and wrote an export package containing the synthetic CSV, validation report, dashboard, and required disclaimer.
    </div>
    <div class="steps">
      {_process_step(1, "Read CSV", f"Loaded {_format_int(real_rows)} real rows for this run.")}
      {_process_step(2, "Mapped fields", _mapping_sentence(mapping))}
      {_process_step(3, "Generated data", f"Created {_format_int(synthetic_rows)} synthetic rows with new event IDs.")}
      {_process_step(4, "Checked patterns", "Compared call timing, call types, priority, geography, and event structure where available.")}
    </div>
  </section>

  <section class="section">
    <h2>Plain-Language Findings</h2>
    {_finding_badges(call_volume_gap, call_type_gap, priority_gap, validation_report, mapping)}
    <p>{escape(_overall_finding(call_volume_gap, call_type_gap, priority_gap, validation_report, mapping))}</p>
  </section>

  <section class="section">
    <h2>Pattern Match Snapshot</h2>
    <p class="subtitle">These are simple match indicators for the most important operational patterns. Smaller gaps mean the synthetic data is closer to the original.</p>
    {_statistical_snapshot(validation_report)}
  </section>

  <section class="section">
    <h2>Privacy Review Checkpoint</h2>
    <p class="subtitle">These screens look for obvious overlap or unusually close records. They support review; they do not certify that a dataset is impossible to re-identify.</p>
    {_privacy_panel(validation_report)}
  </section>

  <section class="section">
    <h2>Operational Analytics</h2>
    <p class="subtitle">These views answer practical questions about frequency, rare categories, and whether the synthetic data keeps the same operational shape.</p>
    {_operational_analytics_panel(real_df, synthetic_df, mapping, geography_column, validation_report)}
  </section>

  <section class="section">
    <h2>Call Volume By Day And Hour</h2>
    <p class="subtitle">Darker cells mean more calls occurred in that day/hour window. The two grids should have a similar shape, even when exact counts differ.</p>
    <div class="heatmaps">
      {_heatmap(real_df, mapping.call_received_datetime, "Real data", "#2563eb")}
      {_heatmap(synthetic_df, mapping.call_received_datetime, "Synthetic data", "#0f766e")}
    </div>
  </section>

  <section class="section">
    <h2>Key Pattern Comparisons</h2>
    <div class="legend">
      <span><span class="swatch" style="background: var(--real);"></span>Real</span>
      <span><span class="swatch" style="background: var(--synthetic);"></span>Synthetic</span>
    </div>
    <div class="grid two">
      {_paired_bar_chart(real_df, synthetic_df, mapping.call_type, "Call Types", 12)}
      {_paired_bar_chart(real_df, synthetic_df, mapping.priority, "Priority Levels", 8)}
    </div>
  </section>

  <section class="section">
    <h2>Geographic Pattern</h2>
    <p class="subtitle">This section compares where calls generally occur using broad area fields, not individual address or coordinate records.</p>
    {_geography_panel(real_df, synthetic_df, mapping, geography_column)}
  </section>

  <section class="section">
    <h2>Response Time</h2>
    {_response_time_panel(real_df, synthetic_df, mapping, validation_report)}
  </section>

  <section class="section">
    <h2>Known Limits Of This Run</h2>
    <ul class="limitations">
      {"".join(f"<li>{escape(item)}</li>" for item in limitations)}
    </ul>
  </section>

  <footer>
    Generated by SyntheticCAD. This dashboard is an executive summary; the researcher validation JSON contains the supporting metrics.
  </footer>
</main>
</body>
</html>
"""


def _metric_card(label: str, value: str, note: str) -> str:
    return f"""
      <div class="metric">
        <div class="label">{escape(label)}</div>
        <div class="value">{escape(value)}</div>
        <div class="note">{escape(note)}</div>
      </div>
    """


def _statistical_snapshot(report: dict[str, Any]) -> str:
    duration = report.get("duration_fields", {}).get("dispatch_to_arrival_minutes", {})
    event_structure = report.get("event_unit_structure", {})
    correlation = report.get("correlation_preservation", {})
    cards = [
        _metric_card(
            "Call timing gap",
            _points_value(report.get("call_volume", {}).get("day_of_week_hour_mean_abs_pct_point_gap")),
            "Average difference across day/hour call-volume cells.",
        ),
        _metric_card(
            "Call type gap",
            _points_value(_largest_gap(report, "call_type")),
            "Largest difference for any call type category.",
        ),
        _metric_card(
            "Disposition gap",
            _points_value(_largest_gap(report, "disposition")),
            "Largest difference for any clearance/disposition category.",
        ),
        _metric_card(
            "Priority match",
            _distance_value(report.get("numeric_fields", {}).get("priority", {}).get("ks_statistic")),
            "Distribution distance. Lower means closer.",
        ),
        _metric_card(
            "Response-time match",
            _distance_value(duration.get("ks_statistic")),
            "Dispatch-to-arrival distribution distance.",
        ),
        _metric_card(
            "Rows per event",
            _distance_value(event_structure.get("rows_per_event_ks_statistic")),
            "Checks whether multi-unit event shape stayed similar.",
        ),
        _metric_card(
            "Relationship gap",
            _distance_value(correlation.get("mean_absolute_correlation_gap")),
            "Average gap across numeric relationships tested.",
        ),
        _metric_card(
            "Relationship direction",
            _percent_value(correlation.get("same_direction_rate")),
            "Share of tested relationships with the same direction.",
        ),
    ]
    return f'<div class="grid metrics">{"".join(cards)}</div>'


def _privacy_panel(report: dict[str, Any]) -> str:
    privacy = report.get("privacy_risk", {})
    if not privacy.get("available"):
        return '<div class="callout">Privacy screens were not available for this run. Review the separate privacy report before sharing the output.</div>'

    exact = privacy.get("exact_quasi_identifier_match", {})
    rare = privacy.get("rare_combination_exposure", {})
    singleton = privacy.get("singleton_category_overlap", {})
    call_type_singleton = singleton.get("Final Call Type", {})
    dcr = privacy.get("distance_to_closest_record", {})
    synthetic_dcr = dcr.get("holdout_to_synthetic", {}).get("median")
    benchmark_dcr = dcr.get("holdout_to_real_train_benchmark", {}).get("median")
    if synthetic_dcr is not None and benchmark_dcr:
        dcr_ratio = synthetic_dcr / benchmark_dcr
    else:
        dcr_ratio = None

    exact_rate = exact.get("synthetic_row_match_rate")
    rare_rate = rare.get("real_rare_presence_rate")
    cards = [
        _metric_card(
            "Full quasi-key overlap",
            _percent_value(exact_rate),
            "Synthetic rows sharing all checked quasi-fields with a real combination.",
        ),
        _metric_card(
            "Rare combination presence",
            _percent_value(rare_rate),
            "Real combinations at or below the rare-category threshold found in synthetic data.",
        ),
        _metric_card(
            "Matching call-type singletons",
            str(call_type_singleton.get("same_singleton_count", "Not checked")),
            "Singleton call types appearing as singletons in both datasets.",
        ),
        _metric_card(
            "Closest-record benchmark",
            f"{dcr_ratio:.2f}x" if dcr_ratio is not None else "Not checked",
            "Synthetic-to-holdout distance compared with the real-to-real benchmark.",
        ),
    ]
    return f'<div class="grid metrics">{"".join(cards)}</div>'


def _operational_analytics_panel(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    geography_column: str | None,
    report: dict[str, Any],
) -> str:
    cards: list[str] = []
    charts: list[str] = []

    if mapping.call_type and mapping.call_type in real_df.columns and mapping.call_type in synthetic_df.columns:
        real_call_counts = _category_counts(real_df[mapping.call_type])
        synthetic_call_counts = _category_counts(synthetic_df[mapping.call_type])
        cards.extend(_frequency_cards("Call type", real_call_counts, synthetic_call_counts))
        charts.append(
            _frequency_spread_chart(
                real_call_counts,
                synthetic_call_counts,
                "Call Type Frequency Spread",
            )
        )

    if geography_column and geography_column in real_df.columns and geography_column in synthetic_df.columns:
        real_geo_counts = _category_counts(real_df[geography_column])
        synthetic_geo_counts = _category_counts(synthetic_df[geography_column])
        cards.extend(_frequency_cards("Area", real_geo_counts, synthetic_geo_counts, limit=2))
        charts.append(
            _frequency_spread_chart(
                real_geo_counts,
                synthetic_geo_counts,
                f"{_geo_title(geography_column)} Frequency Spread",
            )
        )

    duration = report.get("duration_fields", {}).get("dispatch_to_arrival_minutes", {})
    if duration.get("real_count") and duration.get("synthetic_count"):
        real_duration = _duration_minutes(real_df, mapping.dispatch_time, mapping.arrival_time)
        synthetic_duration = _duration_minutes(synthetic_df, mapping.dispatch_time, mapping.arrival_time)
        cards.append(
            _metric_card(
                "90th percentile response",
                f"{_minutes_value(real_duration.quantile(0.90))} / {_minutes_value(synthetic_duration.quantile(0.90))}",
                "Real / synthetic dispatch-to-arrival time for slower calls.",
            )
        )

    event_structure = report.get("event_unit_structure", {})
    if event_structure.get("available"):
        cards.append(
            _metric_card(
                "Events compared",
                f"{_format_int(event_structure.get('real_event_count'))} / {_format_int(event_structure.get('synthetic_event_count'))}",
                "Real / synthetic unique event counts.",
            )
        )

    if not cards and not charts:
        return _empty_chart("Operational Analytics", "No mapped fields were available for frequency analytics.")

    cards_html = f'<div class="grid metrics">{"".join(cards[:8])}</div>' if cards else ""
    charts_html = f'<div class="grid two">{"".join(charts[:2])}</div>' if charts else ""
    return cards_html + charts_html


def _category_counts(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).value_counts()


def _frequency_cards(
    label: str,
    real_counts: pd.Series,
    synthetic_counts: pd.Series,
    limit: int = 3,
) -> list[str]:
    if real_counts.empty and synthetic_counts.empty:
        return []

    cards = [
        _metric_card(
            f"{label} categories",
            f"{_format_int(real_counts.shape[0])} / {_format_int(synthetic_counts.shape[0])}",
            "Real / synthetic distinct non-empty values.",
        )
    ]

    if not real_counts.empty:
        top_category = str(real_counts.index[0])
        cards.append(
            _metric_card(
                f"Most frequent {label.lower()}",
                _short_label(top_category),
                (
                    f"Real {_frequency_note(real_counts, top_category)}; "
                    f"synthetic {_frequency_note(synthetic_counts, top_category)}."
                ),
            )
        )

        real_min = int(real_counts.min())
        synthetic_min = int(synthetic_counts.min()) if not synthetic_counts.empty else 0
        cards.append(
            _metric_card(
                f"Smallest non-zero {label.lower()} frequency",
                f"{_format_int(real_min)} / {_format_int(synthetic_min)}",
                "Real / synthetic minimum category count.",
            )
        )

    return cards[:limit]


def _short_label(value: str, max_length: int = 32) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _frequency_note(counts: pd.Series, category: str) -> str:
    count = int(counts.get(category, 0))
    total = int(counts.sum())
    pct = 100 * count / total if total else 0
    return f"{_format_int(count)} rows ({pct:.1f}%)"


def _frequency_spread_chart(
    real_counts: pd.Series,
    synthetic_counts: pd.Series,
    title: str,
) -> str:
    if real_counts.empty and synthetic_counts.empty:
        return _empty_chart(title, "No category counts were available.")

    bands = [
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-20", 6, 20),
        ("21-100", 21, 100),
        ("101-500", 101, 500),
        ("501-1k", 501, 1000),
        ("1k-5k", 1001, 5000),
        ("5k+", 5001, None),
    ]

    real_band_counts = _band_category_counts(real_counts, bands)
    synthetic_band_counts = _band_category_counts(synthetic_counts, bands)
    max_count = max(max(real_band_counts), max(synthetic_band_counts), 1)

    rows = []
    for index, (label, _, _) in enumerate(bands):
        real_value = real_band_counts[index]
        synthetic_value = synthetic_band_counts[index]
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{escape(label)} rows</div>
              <div class="compare-bars">
                <div class="mini-row">
                  <div class="series-name real">Real</div>
                  <div class="compare-track"><div class="compare-fill real" style="width: {100 * real_value / max_count:.1f}%;"></div></div>
                  <div class="mini-value">{_format_int(real_value)}</div>
                </div>
                <div class="mini-row">
                  <div class="series-name synthetic">Synthetic</div>
                  <div class="compare-track"><div class="compare-fill synthetic" style="width: {100 * synthetic_value / max_count:.1f}%;"></div></div>
                  <div class="mini-value">{_format_int(synthetic_value)}</div>
                </div>
              </div>
              <div class="bar-value"><span class="gap-value">{abs(real_value - synthetic_value)}</span></div>
            </div>
            """
        )

    return f"""
      <div class="chart-panel">
        <h3>{escape(title)}</h3>
        <p class="stat-note">This shows how many categories are rare or common. For example, "1 row" means categories that appeared once.</p>
        {"".join(rows)}
      </div>
    """


def _band_category_counts(
    counts: pd.Series,
    bands: list[tuple[str, int, int | None]],
) -> list[int]:
    values = counts.astype(int)
    output = []
    for _, low, high in bands:
        if high is None:
            output.append(int((values >= low).sum()))
        else:
            output.append(int(((values >= low) & (values <= high)).sum()))
    return output


def _points_value(value: Any) -> str:
    if value is None:
        return "Not checked"
    try:
        return f"{float(value):.2f} pts"
    except (TypeError, ValueError):
        return str(value)


def _distance_value(value: Any) -> str:
    if value is None:
        return "Not checked"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _percent_value(value: Any) -> str:
    if value is None:
        return "Not checked"
    try:
        return f"{100 * float(value):.0f}%"
    except (TypeError, ValueError):
        return str(value)


def _process_step(number: int, title: str, text: str) -> str:
    return f"""
      <div class="step">
        <div class="step-number">{number}</div>
        <h3>{escape(title)}</h3>
        <p>{escape(text)}</p>
      </div>
    """


def _format_int(value: int | float | str | None) -> str:
    if value is None:
        return "Not available"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _source_note(source_limit: Any, source_rows: Any) -> str:
    if source_limit:
        return f"Development run limited to the first {_format_int(source_rows)} source rows."
    return "All loaded rows were used for this run."


def _requested_note(requested_events: Any) -> str:
    if requested_events:
        return f"Requested synthetic event count: {_format_int(requested_events)}."
    return "Synthetic size followed the configured generation settings."


def _field_value(mapped_columns_used: Any, source_columns_found: Any, synthetic_columns: int) -> str:
    if source_columns_found:
        return f"{_format_int(mapped_columns_used)} of {_format_int(source_columns_found)} source fields"
    return f"{_format_int(mapped_columns_used)} mapped / {_format_int(synthetic_columns)} synthetic"


def _field_note(source_columns_found: Any) -> str:
    if source_columns_found:
        return "Only mapped, approved fields are loaded, compared, and exported."
    return "Mapped fields used for this validation run."


def _gap_note(gap: float | None) -> str:
    if gap is None:
        return "Call volume comparison was not available."
    return f"Average day/hour gap: {gap:.2f} percentage points."


def _quality_label(gap: float | None) -> str:
    if gap is None:
        return "Not checked"
    if gap <= 1:
        return "Very close"
    if gap <= 3:
        return "Close"
    return "Needs review"


def _largest_gap(report: dict[str, Any], canonical_name: str) -> float | None:
    field = report.get("categorical_fields", {}).get(canonical_name)
    if not field:
        return None
    return field.get("largest_pct_point_gap")


def _plain_language_summary(
    real_rows: int,
    synthetic_rows: int,
    call_volume_gap: float | None,
    call_type_gap: float | None,
    priority_gap: float | None,
    mapping: SyntheticCADMapping,
) -> str:
    match = _quality_label(call_volume_gap).lower()
    pieces = [
        f"This run used {_format_int(real_rows)} real rows to create {_format_int(synthetic_rows)} synthetic rows.",
        f"The synthetic data produced a {match} match to the original call-volume pattern.",
    ]
    if call_type_gap is not None:
        pieces.append(f"Call type mix differed by at most {call_type_gap:.2f} percentage points for any single category.")
    if priority_gap is not None:
        pieces.append(f"Priority mix differed by at most {priority_gap:.2f} percentage points for any single priority level.")
    if not mapping.dispatch_time or not mapping.arrival_time:
        pieces.append("Response-time matching was not evaluated because this source file does not include dispatch and arrival timestamps.")
    return " ".join(pieces)


def _mapping_sentence(mapping: SyntheticCADMapping) -> str:
    mapped = []
    if mapping.event_id:
        mapped.append("event ID")
    if mapping.call_received_datetime:
        mapped.append("call time")
    if mapping.call_type:
        mapped.append("call type")
    if mapping.priority:
        mapped.append("priority")
    if mapping.location:
        mapped.append("location")
    if not mapped:
        return "No canonical fields were mapped."
    return "Mapped " + ", ".join(mapped) + "."


def _finding_badges(
    call_volume_gap: float | None,
    call_type_gap: float | None,
    priority_gap: float | None,
    report: dict[str, Any],
    mapping: SyntheticCADMapping,
) -> str:
    badges = [
        _badge(f"Call volume: {_quality_label(call_volume_gap)}", call_volume_gap),
    ]
    if call_type_gap is not None:
        badges.append(_badge(f"Call types: max gap {call_type_gap:.2f} pts", call_type_gap, good=5, warn=10))
    if priority_gap is not None:
        badges.append(_badge(f"Priority: max gap {priority_gap:.2f} pts", priority_gap, good=5, warn=10))
    event_structure = report.get("event_unit_structure", {})
    if event_structure.get("available"):
        real_mean = event_structure.get("real_rows_per_event_mean")
        synthetic_mean = event_structure.get("synthetic_rows_per_event_mean")
        badges.append(_badge(f"Rows per event: {real_mean} real / {synthetic_mean} synthetic", 0))
    if not mapping.dispatch_time or not mapping.arrival_time:
        badges.append('<span class="badge warn">Response time: not available</span>')
    return "<p>" + "".join(badges) + "</p>"


def _badge(text: str, value: float | None, good: float = 1, warn: float = 3) -> str:
    class_name = "badge"
    if value is None:
        class_name = "badge warn"
    elif value > warn:
        class_name = "badge stop"
    elif value > good:
        class_name = "badge warn"
    return f'<span class="{class_name}">{escape(text)}</span>'


def _overall_finding(
    call_volume_gap: float | None,
    call_type_gap: float | None,
    priority_gap: float | None,
    report: dict[str, Any],
    mapping: SyntheticCADMapping,
) -> str:
    finding = (
        "The synthetic output is suitable for an early executive review of event-level call patterns. "
        "It preserves the broad shape of when calls occur and the mix of major call categories and priorities."
    )
    if not mapping.unit_id:
        finding += (
            " This source file appears to be one row per call rather than one row per responding unit, "
            "so unit-level staffing patterns were not tested."
        )
    if not mapping.dispatch_time or not mapping.arrival_time:
        finding += " Response-time performance was not tested because the source file does not include the needed timestamps."
    return finding


def _limitations(report: dict[str, Any], mapping: SyntheticCADMapping) -> list[str]:
    limits = [
        "This is an MVP statistical validation view, not a final privacy certification.",
        "Synthetic event IDs are newly generated and do not represent real incidents.",
    ]
    if not mapping.unit_id:
        limits.append("No unit ID field was mapped, so the dashboard cannot evaluate unit-level response patterns.")
    if not mapping.dispatch_time or not mapping.arrival_time:
        limits.append("Dispatch-to-arrival response time cannot be evaluated without dispatch and arrival timestamp fields.")
    if not mapping.latitude or not mapping.longitude:
        limits.append("Latitude and longitude were not available in this source file; geography is compared using broader area fields.")
    duration_available = any(
        item.get("real_count", 0) and item.get("synthetic_count", 0)
        for item in report.get("duration_fields", {}).values()
    )
    if not duration_available:
        limits.append("No response-time distribution chart is shown because this dataset has no usable response-time duration fields.")
    return limits


def _heatmap(df: pd.DataFrame, time_column: str | None, title: str, color: str) -> str:
    if not time_column or time_column not in df.columns:
        return _empty_chart(title, "No call time field was mapped.")

    timestamps = parse_datetime(df[time_column]).dropna()
    if timestamps.empty:
        return _empty_chart(title, "No parseable call times were found.")

    counts = timestamps.groupby([timestamps.dt.dayofweek, timestamps.dt.hour]).size()
    max_count = max(int(counts.max()), 1)
    cells = ['<div></div>']
    for hour in range(24):
        cells.append(f'<div class="hour">{hour if hour in {0, 6, 12, 18, 23} else ""}</div>')
    for day_index, day_label in enumerate(DAY_LABELS):
        cells.append(f'<div class="day">{day_label}</div>')
        for hour in range(24):
            value = int(counts.get((day_index, hour), 0))
            alpha = 0.08 + 0.82 * (value / max_count)
            cells.append(
                f'<div class="cell" title="{escape(day_label)} {hour}:00 - {_format_int(value)} calls" '
                f'style="background: {color}; opacity: {alpha:.3f};"></div>'
            )
    return f"""
      <div class="chart-panel">
        <div class="heatmap-title">{escape(title)}</div>
        <div class="heatmap">{"".join(cells)}</div>
      </div>
    """


def _paired_bar_chart(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    column: str | None,
    title: str,
    limit: int,
) -> str:
    if not column or column not in real_df.columns or column not in synthetic_df.columns:
        return _empty_chart(title, "This field was not available for comparison.")

    real_share = real_df[column].dropna().astype(str).value_counts(normalize=True)
    synthetic_share = synthetic_df[column].dropna().astype(str).value_counts(normalize=True)
    if real_share.empty and synthetic_share.empty:
        return _empty_chart(title, "No values were available for this field.")

    top: list[str] = []
    for category in list(real_share.head(limit).index) + list(synthetic_share.head(limit).index):
        if category not in top:
            top.append(category)
        if len(top) >= limit:
            break
    max_share = max(
        float(real_share.reindex(top, fill_value=0).max()),
        float(synthetic_share.reindex(top, fill_value=0).max()),
        0.01,
    )

    rows = []
    for category in top:
        real_pct = 100 * float(real_share.get(category, 0))
        synthetic_pct = 100 * float(synthetic_share.get(category, 0))
        real_width = 100 * (real_pct / (100 * max_share))
        synthetic_width = 100 * (synthetic_pct / (100 * max_share))
        gap = abs(real_pct - synthetic_pct)
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{escape(str(category))}</div>
              <div class="compare-bars">
                <div class="mini-row">
                  <div class="series-name real">Real</div>
                  <div class="compare-track"><div class="compare-fill real" style="width: {real_width:.1f}%;"></div></div>
                  <div class="mini-value">{real_pct:.1f}%</div>
                </div>
                <div class="mini-row">
                  <div class="series-name synthetic">Synthetic</div>
                  <div class="compare-track"><div class="compare-fill synthetic" style="width: {synthetic_width:.1f}%;"></div></div>
                  <div class="mini-value">{synthetic_pct:.1f}%</div>
                </div>
              </div>
              <div class="bar-value"><span class="gap-value">{gap:.1f} pts</span></div>
            </div>
            """
        )

    return f"""
      <div class="chart-panel">
        <h3>{escape(title)}</h3>
        <p class="stat-note">Each Real and Synthetic bar starts at zero. The pill shows the absolute gap.</p>
        {"".join(rows)}
      </div>
    """


def _empty_chart(title: str, message: str) -> str:
    return f"""
      <div class="chart-panel">
        <h3>{escape(title)}</h3>
        <p class="subtitle">{escape(message)}</p>
      </div>
    """


def _select_geography_column(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    mapping: SyntheticCADMapping,
) -> str | None:
    preferred = [
        "Dispatch Neighborhood",
        "Dispatch Precinct",
        "Dispatch Sector",
        "Dispatch Beat",
        "Neighborhood",
        "PoliceDistrict",
        "district",
        "ZIPCode",
        "Community_Statistical_Areas",
        "PolicePost",
    ]
    for column in preferred:
        if column in real_df.columns and column in synthetic_df.columns:
            return column
    if mapping.latitude and mapping.longitude:
        return None
    if mapping.location and mapping.location in real_df.columns and mapping.location in synthetic_df.columns:
        return mapping.location
    return None


def _geo_title(column: str | None) -> str:
    if not column:
        return "Geography"
    labels = {
        "ZIPCode": "ZIP Code",
        "Community_Statistical_Areas": "Community Statistical Areas",
        "Dispatch Neighborhood": "Dispatch Neighborhood",
        "Dispatch Precinct": "Dispatch Precinct",
        "Dispatch Sector": "Dispatch Sector",
        "Dispatch Beat": "Dispatch Beat",
    }
    return labels.get(column, column)


def _geography_panel(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    geography_column: str | None,
) -> str:
    if geography_column:
        return _paired_bar_chart(
            real_df,
            synthetic_df,
            geography_column,
            f"Call Share By {_geo_title(geography_column)}",
            12,
        )

    return _empty_chart(
        "Geography",
        "No broad geography field was available for comparison. Coordinate density is intentionally excluded from the executive dashboard.",
    )


def _has_coordinates(df: pd.DataFrame, mapping: SyntheticCADMapping) -> bool:
    if not mapping.latitude or not mapping.longitude:
        return False
    if mapping.latitude not in df.columns or mapping.longitude not in df.columns:
        return False
    lat = pd.to_numeric(df[mapping.latitude], errors="coerce").dropna()
    lon = pd.to_numeric(df[mapping.longitude], errors="coerce").dropna()
    return not lat.empty and not lon.empty


def _coordinate_grid(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    title: str,
    color: str,
    bins: int = 12,
    bounds: tuple[float, float, float, float] | None = None,
    max_count: int | None = None,
) -> str:
    if not mapping.latitude or not mapping.longitude:
        return _empty_chart(title, "No coordinate fields were mapped.")

    coords = pd.DataFrame(
        {
            "lat": pd.to_numeric(df[mapping.latitude], errors="coerce"),
            "lon": pd.to_numeric(df[mapping.longitude], errors="coerce"),
        }
    ).dropna()
    if coords.empty:
        return _empty_chart(title, "No usable coordinates were found.")

    if bounds:
        lat_min, lat_max, lon_min, lon_max = bounds
    else:
        lat_min = float(coords["lat"].min())
        lat_max = float(coords["lat"].max())
        lon_min = float(coords["lon"].min())
        lon_max = float(coords["lon"].max())
    if lat_min == lat_max:
        lat_max = lat_min + 0.001
    if lon_min == lon_max:
        lon_max = lon_min + 0.001

    lat_step = (lat_max - lat_min) / bins
    lon_step = (lon_max - lon_min) / bins
    counts: dict[tuple[int, int], int] = {}
    for lat, lon in coords[["lat", "lon"]].itertuples(index=False):
        row = int((lat - lat_min) / lat_step)
        col = int((lon - lon_min) / lon_step)
        row = max(0, min(bins - 1, row))
        col = max(0, min(bins - 1, col))
        row = bins - 1 - row
        counts[(row, col)] = counts.get((row, col), 0) + 1

    scale_max = max_count or (max(counts.values()) if counts else 1)
    scale_max = max(scale_max, 1)
    cells = []
    for row in range(bins):
        for col in range(bins):
            value = counts.get((row, col), 0)
            alpha = 0.08 + 0.82 * (value / scale_max) if value else 0.08
            cells.append(
                f'<div class="geo-cell" title="{_format_int(value)} calls in this area" '
                f'style="background: {color}; opacity: {alpha:.3f};"></div>'
            )

    return f"""
      <div class="chart-panel">
        <div class="heatmap-title">{escape(title)}</div>
        <div class="geo-grid">{"".join(cells)}</div>
      </div>
    """


def _coordinate_max_count(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    bins: int = 12,
    bounds: tuple[float, float, float, float] | None = None,
) -> int:
    return max(
        _coordinate_grid_max(real_df, mapping, bins=bins, bounds=bounds),
        _coordinate_grid_max(synthetic_df, mapping, bins=bins, bounds=bounds),
        1,
    )


def _coordinate_grid_max(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    bins: int = 12,
    bounds: tuple[float, float, float, float] | None = None,
) -> int:
    coords = _coordinate_frame(df, mapping)
    if coords.empty:
        return 0
    if bounds:
        lat_min, lat_max, lon_min, lon_max = bounds
    else:
        lat_min = float(coords["lat"].min())
        lat_max = float(coords["lat"].max())
        lon_min = float(coords["lon"].min())
        lon_max = float(coords["lon"].max())
    if lat_min == lat_max:
        lat_max = lat_min + 0.001
    if lon_min == lon_max:
        lon_max = lon_min + 0.001
    lat_step = (lat_max - lat_min) / bins
    lon_step = (lon_max - lon_min) / bins
    counts: dict[tuple[int, int], int] = {}
    for lat, lon in coords[["lat", "lon"]].itertuples(index=False):
        row = int((lat - lat_min) / lat_step)
        col = int((lon - lon_min) / lon_step)
        row = max(0, min(bins - 1, row))
        col = max(0, min(bins - 1, col))
        row = bins - 1 - row
        counts[(row, col)] = counts.get((row, col), 0) + 1
    return max(counts.values()) if counts else 0


def _coordinate_bounds(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    mapping: SyntheticCADMapping,
) -> tuple[float, float, float, float]:
    real_coords = _coordinate_frame(real_df, mapping)
    synthetic_coords = _coordinate_frame(synthetic_df, mapping)
    coords = pd.concat([real_coords, synthetic_coords], ignore_index=True).dropna()
    lat_min = float(coords["lat"].min())
    lat_max = float(coords["lat"].max())
    lon_min = float(coords["lon"].min())
    lon_max = float(coords["lon"].max())
    if lat_min == lat_max:
        lat_max = lat_min + 0.001
    if lon_min == lon_max:
        lon_max = lon_min + 0.001
    return lat_min, lat_max, lon_min, lon_max


def _coordinate_frame(df: pd.DataFrame, mapping: SyntheticCADMapping) -> pd.DataFrame:
    if not mapping.latitude or not mapping.longitude:
        return pd.DataFrame({"lat": [], "lon": []})
    return pd.DataFrame(
        {
            "lat": pd.to_numeric(df[mapping.latitude], errors="coerce"),
            "lon": pd.to_numeric(df[mapping.longitude], errors="coerce"),
        }
    ).dropna()


def _response_time_panel(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    report: dict[str, Any],
) -> str:
    duration = report.get("duration_fields", {}).get("dispatch_to_arrival_minutes", {})
    if not duration.get("real_count") or not duration.get("synthetic_count"):
        return """
          <div class="chart-panel">
            <h3>Dispatch-To-Arrival Time</h3>
            <p class="subtitle">Not available for this dataset. A response-time chart requires separate dispatch and arrival timestamp fields.</p>
          </div>
        """

    real_duration = _duration_minutes(real_df, mapping.dispatch_time, mapping.arrival_time)
    synthetic_duration = _duration_minutes(synthetic_df, mapping.dispatch_time, mapping.arrival_time)
    real_median = duration.get("real_median")
    synthetic_median = duration.get("synthetic_median")
    median_gap = None
    if real_median is not None and synthetic_median is not None:
        median_gap = abs(float(real_median) - float(synthetic_median))

    cards = [
        _metric_card("Real median", _minutes_value(real_median), "Middle dispatch-to-arrival time in the real data."),
        _metric_card("Synthetic median", _minutes_value(synthetic_median), "Middle dispatch-to-arrival time in the synthetic data."),
        _metric_card("Median gap", _minutes_value(median_gap), "Absolute difference between the two medians."),
        _metric_card("Distribution distance", _distance_value(duration.get("ks_statistic")), "Lower means the full response-time pattern is closer."),
    ]
    return f"""
      <div class="grid metrics">{"".join(cards)}</div>
      {_duration_curve(real_duration, synthetic_duration, "Dispatch-To-Arrival Distribution Curve")}
      {_duration_histogram(real_duration, synthetic_duration, "Dispatch-To-Arrival Buckets")}
    """


def _duration_minutes(
    df: pd.DataFrame,
    start_column: str | None,
    end_column: str | None,
) -> pd.Series:
    if not start_column or not end_column:
        return pd.Series(dtype=float)
    if start_column not in df.columns or end_column not in df.columns:
        return pd.Series(dtype=float)
    start = parse_datetime(df[start_column])
    end = parse_datetime(df[end_column])
    minutes = (end - start).dt.total_seconds() / 60
    return minutes[(minutes >= 0) & (minutes <= 24 * 60)].dropna()


def _minutes_value(value: Any) -> str:
    if value is None:
        return "Not checked"
    try:
        return f"{float(value):.2f} min"
    except (TypeError, ValueError):
        return str(value)


def _duration_curve(real: pd.Series, synthetic: pd.Series, title: str) -> str:
    if real.empty or synthetic.empty:
        return _empty_chart(title, "No usable duration values were available.")

    max_value = max(float(real.quantile(0.95)), float(synthetic.quantile(0.95)), 1)
    x_values = [max_value * index / 30 for index in range(31)]
    real_points = [(x, 100 * float((real <= x).mean())) for x in x_values]
    synthetic_points = [(x, 100 * float((synthetic <= x).mean())) for x in x_values]

    width = 760
    height = 260
    left = 54
    right = 18
    top = 18
    bottom = 42
    chart_width = width - left - right
    chart_height = height - top - bottom

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, pct = point
        px = left + chart_width * (x / max_value)
        py = top + chart_height * (1 - pct / 100)
        return px, py

    def polyline(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in [project(point) for point in points])

    x_ticks = [0, max_value / 2, max_value]
    y_ticks = [0, 50, 100]
    grid = []
    for pct in y_ticks:
        y = top + chart_height * (1 - pct / 100)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e6ebf0" />'
        )
        grid.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" fill="#5d6875" font-size="11">{pct}%</text>'
        )
    for tick in x_ticks:
        x = left + chart_width * (tick / max_value)
        grid.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" stroke="#eef2f6" />'
        )
        grid.append(
            f'<text x="{x:.1f}" y="{height - 17}" text-anchor="middle" fill="#5d6875" font-size="11">{tick:.0f} min</text>'
        )

    return f"""
      <div class="curve-card">
        <h3>{escape(title)}</h3>
        <p class="stat-note">This curve shows the share of calls that arrived within each number of minutes. The blue and green lines should closely track each other.</p>
        <svg class="curve-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">
          <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />
          {"".join(grid)}
          <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#b8c1cc" />
          <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#b8c1cc" />
          <polyline points="{polyline(real_points)}" fill="none" stroke="#2563eb" stroke-width="3" />
          <polyline points="{polyline(synthetic_points)}" fill="none" stroke="#0f766e" stroke-width="3" />
          <circle cx="{project(real_points[-1])[0]:.1f}" cy="{project(real_points[-1])[1]:.1f}" r="4" fill="#2563eb" />
          <circle cx="{project(synthetic_points[-1])[0]:.1f}" cy="{project(synthetic_points[-1])[1]:.1f}" r="4" fill="#0f766e" />
          <text x="{left}" y="14" fill="#2563eb" font-size="12" font-weight="700">Real</text>
          <text x="{left + 46}" y="14" fill="#0f766e" font-size="12" font-weight="700">Synthetic</text>
          <text x="{width / 2:.1f}" y="{height - 2}" text-anchor="middle" fill="#5d6875" font-size="11">Minutes from dispatch to arrival</text>
        </svg>
      </div>
    """


def _duration_histogram(real: pd.Series, synthetic: pd.Series, title: str) -> str:
    if real.empty or synthetic.empty:
        return _empty_chart(title, "No usable duration values were available.")
    max_value = max(float(real.quantile(0.95)), float(synthetic.quantile(0.95)), 1)
    bins = pd.interval_range(start=0, end=max_value, periods=10)
    real_counts = pd.cut(real.clip(upper=max_value), bins=bins, include_lowest=True).value_counts(normalize=True).sort_index()
    synthetic_counts = pd.cut(synthetic.clip(upper=max_value), bins=bins, include_lowest=True).value_counts(normalize=True).sort_index()
    rows = []
    max_share = max(float(real_counts.max()), float(synthetic_counts.max()), 0.01)
    for interval in bins:
        label = f"{interval.left:.0f}-{interval.right:.0f} min"
        real_pct = 100 * float(real_counts.get(interval, 0))
        synthetic_pct = 100 * float(synthetic_counts.get(interval, 0))
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{escape(label)}</div>
              <div class="compare-bars">
                <div class="mini-row">
                  <div class="series-name real">Real</div>
                  <div class="compare-track"><div class="compare-fill real" style="width: {100 * real_pct / (100 * max_share):.1f}%;"></div></div>
                  <div class="mini-value">{real_pct:.1f}%</div>
                </div>
                <div class="mini-row">
                  <div class="series-name synthetic">Synthetic</div>
                  <div class="compare-track"><div class="compare-fill synthetic" style="width: {100 * synthetic_pct / (100 * max_share):.1f}%;"></div></div>
                  <div class="mini-value">{synthetic_pct:.1f}%</div>
                </div>
              </div>
              <div class="bar-value"><span class="gap-value">{abs(real_pct - synthetic_pct):.1f} pts</span></div>
            </div>
            """
        )
    return f"""
      <div class="chart-panel">
        <h3>{escape(title)}</h3>
        <p class="stat-note">Each bucket shows the share of calls in that response-time range. The pill shows the real/synthetic gap.</p>
        {"".join(rows)}
      </div>
    """
