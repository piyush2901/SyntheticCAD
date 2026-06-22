"""Human-friendly field mapping guide for CSV profiling."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from syntheticcad.schema import FIELD_DEFINITIONS, SyntheticCADMapping


REQUIRED_FIELD_ORDER = [
    "event_id",
    "call_received_datetime",
    "location",
    "latitude",
    "longitude",
    "call_type",
    "priority",
    "disposition",
    "unit_id",
    "dispatch_time",
    "arrival_time",
    "clearance_time",
]


def write_mapping_guide(
    profile: dict[str, Any],
    out_path: str | Path,
    mapping: SyntheticCADMapping | None = None,
) -> Path:
    """Write a standalone HTML guide for reviewing suggested field mappings."""

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_mapping = mapping or SyntheticCADMapping.from_dict(
        profile.get("suggested_mapping", {})
    )
    html = _render_mapping_guide(profile, resolved_mapping)
    target.write_text(html, encoding="utf-8")
    return target


def _render_mapping_guide(profile: dict[str, Any], mapping: SyntheticCADMapping) -> str:
    shape = profile.get("shape", {})
    row_count = shape.get("rows", 0)
    column_count = shape.get("columns", 0)
    diagnostics = profile.get("event_unit_diagnostics", {})
    event_note = _event_unit_note(diagnostics, mapping)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SyntheticCAD Field Mapping Guide</title>
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
      padding: 24px 0 18px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 18px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 19px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; letter-spacing: 0; }}
    p {{ margin: 0 0 10px; }}
    code {{
      background: #e9eef5;
      border-radius: 4px;
      padding: 2px 5px;
      font-family: Consolas, monospace;
      font-size: 13px;
    }}
    .subtitle {{
      color: var(--muted);
      max-width: 900px;
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
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .metric {{
      border: 1px solid var(--line);
      background: var(--soft);
      border-radius: 8px;
      padding: 14px;
      min-height: 98px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric .value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .callout {{
      border-left: 4px solid var(--blue);
      background: #f7faff;
      padding: 12px 14px;
      border-radius: 6px;
    }}
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
      margin-bottom: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      background: #fff;
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    th {{
      background: #f1f4f8;
      color: #2f3b49;
      font-size: 12px;
      text-transform: uppercase;
    }}
    .status {{
      display: inline-flex;
      border-radius: 999px;
      padding: 4px 8px;
      color: #fff;
      font-weight: 700;
      font-size: 12px;
      white-space: nowrap;
    }}
    .ok {{ background: var(--green); }}
    .review {{ background: var(--amber); }}
    .missing {{ background: var(--red); }}
    .level {{
      color: var(--muted);
      font-weight: 700;
      text-transform: capitalize;
    }}
    .samples {{
      color: var(--muted);
      font-size: 12px;
    }}
    .columns {{
      columns: 3;
      color: var(--muted);
      font-size: 13px;
      margin: 0;
      padding-left: 18px;
    }}
    @media (max-width: 900px) {{
      main {{ padding: 18px; }}
      .metrics, .steps {{ grid-template-columns: 1fr; }}
      .columns {{ columns: 1; }}
      table {{ table-layout: auto; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>SyntheticCAD Field Mapping Guide</h1>
    <p class="subtitle">Use this review page to confirm which columns in the uploaded CSV correspond to the fields SyntheticCAD needs. The app can suggest likely matches, but the user should confirm the mapping before generation.</p>
  </header>

  <section class="section">
    <h2>CSV Summary</h2>
    <div class="grid metrics">
      {_metric_card("Rows profiled", _format_int(row_count))}
      {_metric_card("Columns found", _format_int(column_count))}
      {_metric_card("Event/unit shape", event_note)}
    </div>
  </section>

  <section class="section">
    <h2>How The User Should Be Guided</h2>
    <div class="callout">
      Start with the suggested matches, then ask the user to confirm only the fields that matter for the first run. If a field is missing, leave it blank and explain which dashboard or validation section will be unavailable.
    </div>
    <div class="steps">
      {_step(1, "Confirm event fields", "Identify the call ID, call time, call type, priority, disposition, and broad location fields.")}
      {_step(2, "Confirm unit fields", "If the export has one row per responding unit, map unit ID, dispatch time, arrival time, and clearance time.")}
      {_step(3, "Check samples", "Use sample values to avoid mapping similar-looking columns to the wrong field.")}
      {_step(4, "Explain gaps", "Tell the user what cannot be validated when a field is not present.")}
    </div>
  </section>

  <section class="section">
    <h2>Suggested Mapping</h2>
    <table>
      <thead>
        <tr>
          <th style="width: 16%;">Expected Field</th>
          <th style="width: 10%;">Level</th>
          <th style="width: 22%;">What It Means</th>
          <th style="width: 18%;">Suggested Column</th>
          <th style="width: 14%;">Status</th>
          <th style="width: 20%;">Sample Values</th>
        </tr>
      </thead>
      <tbody>
        {"".join(_field_row(profile, mapping, field_name) for field_name in REQUIRED_FIELD_ORDER)}
      </tbody>
    </table>
  </section>

  <section class="section">
    <h2>Event-Level Vs Unit-Level</h2>
    <p>Event-level fields describe the call itself and usually repeat when more than one unit responds. Unit-level fields describe a specific responding officer, car, or unit and can differ across rows with the same Event ID.</p>
    <p>{escape(_event_unit_detail(diagnostics, mapping))}</p>
  </section>

  <section class="section">
    <h2>Columns Found In This CSV</h2>
    <ul class="columns">
      {"".join(f"<li><code>{escape(column)}</code></li>" for column in profile.get("columns", []))}
    </ul>
  </section>
</main>
</body>
</html>
"""


def _metric_card(label: str, value: str) -> str:
    return f"""
      <div class="metric">
        <div class="label">{escape(label)}</div>
        <div class="value">{escape(value)}</div>
      </div>
    """


def _step(number: int, title: str, text: str) -> str:
    return f"""
      <div class="step">
        <div class="step-number">{number}</div>
        <h3>{escape(title)}</h3>
        <p>{escape(text)}</p>
      </div>
    """


def _field_row(profile: dict[str, Any], mapping: SyntheticCADMapping, field_name: str) -> str:
    definition = FIELD_DEFINITIONS[field_name]
    column = getattr(mapping, field_name)
    column_profiles = profile.get("column_profiles", {})
    suggestion = profile.get("mapping_suggestions", {}).get(field_name, {})
    confidence = suggestion.get("confidence", 0)
    samples = []
    if column and column in column_profiles:
        samples = column_profiles[column].get("sample_values", [])

    status_class, status_label = _status(column, confidence)
    sample_text = ", ".join(str(item) for item in samples[:4]) if samples else "No sample values"
    suggested = f"<code>{escape(column)}</code>" if column else "No suggestion"

    return f"""
      <tr>
        <td><strong>{escape(definition["label"])}</strong></td>
        <td><span class="level">{escape(definition["level"])}</span></td>
        <td>{escape(definition["description"])}</td>
        <td>{suggested}</td>
        <td><span class="status {status_class}">{escape(status_label)}</span></td>
        <td><span class="samples">{escape(sample_text)}</span></td>
      </tr>
    """


def _status(column: str | None, confidence: float) -> tuple[str, str]:
    if not column:
        return "missing", "Not found"
    if confidence >= 0.8:
        return "ok", "Likely match"
    return "review", "Review"


def _event_unit_note(diagnostics: dict[str, Any], mapping: SyntheticCADMapping) -> str:
    if not diagnostics.get("available"):
        return "Needs Event ID"
    rows = diagnostics.get("rows_per_event", {})
    pct_multi = rows.get("pct_events_with_multiple_rows", 0)
    if pct_multi and pct_multi >= 5:
        return "Multi-unit likely"
    return "Mostly one row per call"


def _event_unit_detail(diagnostics: dict[str, Any], mapping: SyntheticCADMapping) -> str:
    if not diagnostics.get("available"):
        return "SyntheticCAD could not evaluate the event/unit structure because no Event ID field was mapped."

    rows = diagnostics.get("rows_per_event", {})
    pct_multi = rows.get("pct_events_with_multiple_rows", 0)
    average = rows.get("mean", 0)
    if pct_multi and pct_multi >= 5:
        return (
            f"The profile found an average of {average} rows per event and {pct_multi}% of events with multiple rows. "
            "That suggests this CSV may include one row per responding unit, so unit-level fields should be reviewed carefully."
        )
    return (
        f"The profile found an average of {average} rows per event and only {pct_multi}% of events with multiple rows. "
        "This dataset appears to be mostly event-level, so response-time and unit-level validation may require a richer CAD export."
    )


def _format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)
