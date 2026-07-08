"""Local browser app for the SyntheticCAD prototype."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import traceback
from typing import Any, Callable
from urllib.parse import parse_qs, quote, unquote, urlparse
import uuid

from syntheticcad.dashboard import write_executive_dashboard
from syntheticcad.mapping_guide import write_mapping_guide
from syntheticcad.profiling import build_profile, read_csv, write_json
from syntheticcad.schema import SyntheticCADMapping, load_mapping, save_mapping
from syntheticcad.synthesis import (
    synthesize_baseline_result,
    synthesize_pattern_matched,
    synthesize_sdv,
    write_export_package,
)
from syntheticcad.validation import validate_synthetic_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 8765
JOBS: dict[str, "JobState"] = {}
JOBS_LOCK = threading.Lock()


@dataclass
class JobState:
    id: str
    kind: str
    status: str = "queued"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with JOBS_LOCK:
            self.logs.append(f"[{timestamp}] {message}")

    def to_dict(self) -> dict[str, Any]:
        elapsed_to = self.finished_at or time.time()
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "elapsed_seconds": round(elapsed_to - self.started_at, 1),
            "logs": self.logs,
            "result": self.result,
            "error": self.error,
        }


def _resolve_path(value: str, must_exist: bool = False) -> Path:
    path = Path(value.strip().strip('"')).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return int(value)


def _form_value(form: dict[str, list[str]], name: str, default: str = "") -> str:
    values = form.get(name, [])
    return values[0] if values else default


def _artifact(path: Path, label: str) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(PROJECT_ROOT.resolve())
        url = "/artifacts/" + quote(str(rel).replace("\\", "/"), safe="/")
    except ValueError:
        url = resolved.as_uri()
    return {
        "label": label,
        "path": str(resolved),
        "url": url,
    }


def _profile_job(params: dict[str, str], job: JobState) -> dict[str, Any]:
    csv_path = _resolve_path(params["csv_path"], must_exist=True)
    max_rows = _optional_int(params.get("profile_max_rows"))
    out_dir = _resolve_path(params["profile_out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    job.log(f"Reading {csv_path.name}")
    df = read_csv(csv_path, nrows=max_rows)
    job.log(f"Profiled {df.shape[0]:,} rows and {df.shape[1]:,} columns")

    profile = build_profile(df)
    profile_path = out_dir / "profile.json"
    mapping_path = out_dir / "mapping_suggested.json"
    guide_path = out_dir / "mapping_guide.html"

    write_json(profile, profile_path)
    save_mapping(SyntheticCADMapping.from_dict(profile["suggested_mapping"]), mapping_path)
    write_mapping_guide(profile, guide_path)
    job.log("Wrote profile, suggested mapping, and mapping guide")

    diagnostics = profile.get("event_unit_diagnostics", {})
    return {
        "summary": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "event_unit_available": bool(diagnostics.get("available")),
            "event_count": diagnostics.get("event_count"),
        },
        "artifacts": [
            _artifact(guide_path, "Mapping Guide"),
            _artifact(mapping_path, "Suggested Mapping JSON"),
            _artifact(profile_path, "Profile JSON"),
        ],
    }


def _synthesize_job(params: dict[str, str], job: JobState) -> dict[str, Any]:
    csv_path = _resolve_path(params["csv_path"], must_exist=True)
    mapping_path = _resolve_path(params["mapping_path"], must_exist=True)
    out_dir = _resolve_path(params["synthesis_out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    method = params.get("method", "pattern")
    seed = int(params.get("seed") or 42)
    source_limit = _optional_int(params.get("source_max_rows"))
    event_count = _optional_int(params.get("events"))

    mapping = load_mapping(mapping_path)
    source_header = read_csv(csv_path, nrows=0)
    source_column_count = int(source_header.shape[1])
    mapped_columns = mapping.mapped_columns()
    missing_columns = [column for column in mapped_columns if column not in source_header.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"The mapping references columns that are not in this CSV: {missing}")

    job.log(f"Reading mapped columns from {csv_path.name}")
    real_df = read_csv(csv_path, usecols=mapped_columns, nrows=source_limit)
    job.log(f"Loaded {real_df.shape[0]:,} rows and {real_df.shape[1]:,} mapped columns")

    if method == "baseline":
        job.log("Generating with baseline engine")
        synthesis = synthesize_baseline_result(real_df, mapping, event_count=event_count, seed=seed)
    elif method == "sdv":
        job.log("Generating with SDV")
        synthesis = synthesize_sdv(real_df, mapping, event_count=event_count, seed=seed)
    else:
        job.log("Generating with empirical pattern matcher")
        synthesis = synthesize_pattern_matched(real_df, mapping, event_count=event_count, seed=seed)

    synthetic_df = synthesis.dataframe
    job.log(f"Generated {synthetic_df.shape[0]:,} synthetic rows")

    validation = validate_synthetic_data(real_df, synthetic_df, mapping)
    run_metadata = {
        "source_file_name": csv_path.name,
        "source_rows_used": int(real_df.shape[0]),
        "source_row_limit": source_limit,
        "source_columns_found": source_column_count,
        "mapped_columns_used": len(mapped_columns),
        "mapped_column_names": mapped_columns,
        "requested_synthetic_events": event_count,
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
            f"Read {real_df.shape[0]:,} real rows from {csv_path.name}, used the "
            f"approved field mapping and {synthesis.library_used}, and generated "
            f"{synthetic_df.shape[0]:,} synthetic rows locally."
        ),
        **run_metadata,
    }

    paths = write_export_package(synthetic_df, out_dir, validation_report=validation)
    if synthesis.method.startswith("sdv") and synthesis.details.get("metadata"):
        metadata_path = out_dir / "sdv_metadata.json"
        write_json(synthesis.details["metadata"], metadata_path)
        paths["sdv_metadata"] = metadata_path

    dashboard_path = write_executive_dashboard(
        real_df,
        synthetic_df,
        mapping,
        validation,
        out_dir / "executive_dashboard.html",
        run_metadata=run_metadata,
    )
    paths["executive_dashboard"] = dashboard_path
    job.log("Wrote export package and dashboard")

    call_volume_gap = validation.get("call_volume", {}).get("day_of_week_hour_mean_abs_pct_point_gap")
    event_unit = validation.get("event_unit_structure", {})
    correlation = validation.get("correlation_preservation", {})
    return {
        "summary": {
            "real_rows": int(real_df.shape[0]),
            "synthetic_rows": int(synthetic_df.shape[0]),
            "method": synthesis.method,
            "library": synthesis.library_used,
            "call_volume_gap": call_volume_gap,
            "real_events": event_unit.get("real_event_count"),
            "synthetic_events": event_unit.get("synthetic_event_count"),
            "mean_correlation_gap": correlation.get("mean_absolute_correlation_gap"),
        },
        "artifacts": [
            _artifact(dashboard_path, "Executive Dashboard"),
            _artifact(paths["synthetic_csv"], "Synthetic CSV"),
            _artifact(paths["validation_report"], "Validation JSON"),
            _artifact(paths["disclaimer"], "Disclaimer"),
        ],
    }


def _start_job(kind: str, params: dict[str, str], runner: Callable[[dict[str, str], JobState], dict[str, Any]]) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = JobState(id=job_id, kind=kind)
    with JOBS_LOCK:
        JOBS[job_id] = job

    def run() -> None:
        with JOBS_LOCK:
            job.status = "running"
        try:
            job.result = runner(params, job)
            with JOBS_LOCK:
                job.status = "completed"
                job.finished_at = time.time()
        except Exception as exc:  # pragma: no cover - local app boundary
            job.log(traceback.format_exc().strip())
            with JOBS_LOCK:
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = time.time()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return job_id


def _default_csv_path() -> str:
    candidate = Path.home() / "Downloads" / "Call_Data_20260619.csv"
    return str(candidate) if candidate.exists() else ""


def _default_output_dir(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("outputs") / f"{prefix}_{stamp}")


def _html_page() -> str:
    csv_path = _default_csv_path()
    profile_out = _default_output_dir("app_profile")
    synth_out = _default_output_dir("app_synthetic")
    mapping_path = str(Path("configs") / "mappings" / "seattle_2025_mvp.json")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SyntheticCAD Local Prototype</title>
  <style>
    :root {{
      --bg: #f6f7f2;
      --panel: #ffffff;
      --text: #17201d;
      --muted: #64706b;
      --line: #d9ded7;
      --accent: #236c64;
      --accent-strong: #174f49;
      --amber: #a8641e;
      --blue: #315f86;
      --danger: #9d2d27;
      --soft: #e9f0ee;
      --shadow: 0 12px 28px rgba(23, 32, 29, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #fbfcf8;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 700;
    }}
    .subhead {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }}
    .pill {{
      border: 1px solid #bdd2ce;
      color: var(--accent-strong);
      background: var(--soft);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      white-space: nowrap;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      max-width: 1440px;
      margin: 0 auto;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .controls {{
      padding: 18px;
    }}
    .workspace {{
      min-height: calc(100vh - 110px);
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 15px;
      font-weight: 700;
    }}
    label {{
      display: block;
      margin: 14px 0 6px;
      color: #2f3a36;
      font-size: 13px;
      font-weight: 650;
    }}
    input, select {{
      width: 100%;
      min-height: 38px;
      border: 1px solid #c9d0c8;
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-size: 13px;
    }}
    input:focus, select:focus {{
      outline: 3px solid rgba(35, 108, 100, 0.15);
      border-color: var(--accent);
    }}
    .grid-two {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .actions {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      margin-top: 18px;
    }}
    button {{
      border: 0;
      border-radius: 6px;
      min-height: 40px;
      padding: 9px 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    button.primary {{ background: var(--accent); color: #fff; }}
    button.primary:hover {{ background: var(--accent-strong); }}
    button.secondary {{ background: #eef1ec; color: #1f2b27; border: 1px solid #cfd7ce; }}
    button.secondary:hover {{ background: #e4e9e2; }}
    button:disabled {{ opacity: 0.55; cursor: not-allowed; }}
    .status-head {{
      padding: 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .status-body {{
      padding: 18px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(280px, 360px);
      gap: 18px;
    }}
    .state {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 86px;
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      color: #fff;
      background: var(--blue);
    }}
    .state.completed {{ background: var(--accent); }}
    .state.failed {{ background: var(--danger); }}
    .state.idle {{ background: #78817d; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 78px;
      background: #fcfdf9;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    .metric .value {{
      margin-top: 8px;
      font-size: 22px;
      font-weight: 800;
      color: #15211d;
      word-break: break-word;
    }}
    .artifact-list {{
      display: grid;
      gap: 10px;
    }}
    .artifact {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      background: #fff;
    }}
    .artifact a {{
      color: var(--blue);
      font-weight: 750;
      text-decoration: none;
    }}
    .artifact a:hover {{ text-decoration: underline; }}
    .artifact small {{
      display: block;
      color: var(--muted);
      margin-top: 3px;
      overflow-wrap: anywhere;
    }}
    pre {{
      margin: 0;
      min-height: 360px;
      max-height: calc(100vh - 320px);
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #101815;
      color: #d9efe7;
      padding: 14px;
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-wrap;
    }}
    .empty {{
      border: 1px dashed #c4ccc2;
      border-radius: 8px;
      padding: 18px;
      color: var(--muted);
      background: #fbfcf8;
    }}
    .note {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      margin-top: 8px;
    }}
    @media (max-width: 980px) {{
      main, .status-body {{ grid-template-columns: 1fr; }}
      .workspace {{ min-height: auto; }}
      .metrics {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>SyntheticCAD Local Prototype</h1>
      <div class="subhead">CAD ingestion, synthesis, validation, and export</div>
    </div>
    <div class="pill">Offline local processing</div>
  </header>
  <main>
    <section class="controls">
      <h2>Run Configuration</h2>
      <form id="app-form">
        <label for="csv_path">CAD CSV path</label>
        <input id="csv_path" name="csv_path" value="{csv_path}" placeholder="C:\\path\\to\\cad.csv">

        <label for="mapping_path">Mapping JSON path</label>
        <input id="mapping_path" name="mapping_path" value="{mapping_path}">

        <label for="method">Synthesis method</label>
        <select id="method" name="method">
          <option value="pattern" selected>Pattern matcher</option>
          <option value="sdv">SDV</option>
          <option value="baseline">Baseline</option>
        </select>

        <div class="grid-two">
          <div>
            <label for="source_max_rows">Source row limit</label>
            <input id="source_max_rows" name="source_max_rows" placeholder="blank = all rows">
          </div>
          <div>
            <label for="events">Synthetic events</label>
            <input id="events" name="events" placeholder="blank = match input">
          </div>
        </div>

        <div class="grid-two">
          <div>
            <label for="seed">Seed</label>
            <input id="seed" name="seed" value="42">
          </div>
          <div>
            <label for="profile_max_rows">Profile rows</label>
            <input id="profile_max_rows" name="profile_max_rows" placeholder="blank = all rows">
          </div>
        </div>

        <label for="synthesis_out_dir">Synthetic output folder</label>
        <input id="synthesis_out_dir" name="synthesis_out_dir" value="{synth_out}">

        <label for="profile_out_dir">Profile output folder</label>
        <input id="profile_out_dir" name="profile_out_dir" value="{profile_out}">

        <div class="actions">
          <button class="primary" id="run-synthesis" type="button">Generate Synthetic Dataset</button>
          <button class="secondary" id="run-profile" type="button">Profile CSV</button>
        </div>
      </form>
      <div class="note">Use the pattern matcher for the fastest full-file MVP run. Use SDV for library-backed synthesis checks.</div>
    </section>
    <section class="workspace">
      <div class="status-head">
        <h2>Run Status</h2>
        <span id="state" class="state idle">Idle</span>
      </div>
      <div class="status-body">
        <div>
          <div id="metrics" class="metrics"></div>
          <pre id="log">Ready.</pre>
        </div>
        <div>
          <h2>Artifacts</h2>
          <div id="artifacts" class="empty">No artifacts yet.</div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const form = document.getElementById('app-form');
    const state = document.getElementById('state');
    const log = document.getElementById('log');
    const metrics = document.getElementById('metrics');
    const artifacts = document.getElementById('artifacts');
    const profileButton = document.getElementById('run-profile');
    const synthButton = document.getElementById('run-synthesis');

    function setBusy(busy) {{
      profileButton.disabled = busy;
      synthButton.disabled = busy;
    }}

    function setState(value) {{
      state.textContent = value;
      state.className = 'state ' + value.toLowerCase();
    }}

    function formBody() {{
      return new URLSearchParams(new FormData(form)).toString();
    }}

    function renderMetrics(result) {{
      const summary = result?.summary || {{}};
      const items = [];
      if (summary.real_rows !== undefined) items.push(['Real rows', summary.real_rows.toLocaleString()]);
      if (summary.synthetic_rows !== undefined) items.push(['Synthetic rows', summary.synthetic_rows.toLocaleString()]);
      if (summary.method) items.push(['Method', summary.method]);
      if (summary.call_volume_gap !== undefined && summary.call_volume_gap !== null) items.push(['Call volume gap', summary.call_volume_gap + ' pts']);
      if (summary.mean_correlation_gap !== undefined && summary.mean_correlation_gap !== null) items.push(['Correlation gap', summary.mean_correlation_gap]);
      if (summary.event_count !== undefined && summary.event_count !== null) items.push(['Events', summary.event_count.toLocaleString()]);
      metrics.innerHTML = items.map(([label, value]) => `
        <div class="metric"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>
      `).join('');
    }}

    function renderArtifacts(result) {{
      const list = result?.artifacts || [];
      if (!list.length) {{
        artifacts.className = 'empty';
        artifacts.textContent = 'No artifacts yet.';
        return;
      }}
      artifacts.className = 'artifact-list';
      artifacts.innerHTML = list.map(item => `
        <div class="artifact">
          <div>
            <a href="${{item.url}}" target="_blank" rel="noreferrer">${{item.label}}</a>
            <small>${{item.path}}</small>
          </div>
        </div>
      `).join('');
    }}

    async function startJob(endpoint) {{
      setBusy(true);
      setState('Running');
      metrics.innerHTML = '';
      artifacts.className = 'empty';
      artifacts.textContent = 'Waiting for outputs.';
      log.textContent = 'Starting...';
      try {{
        const response = await fetch(endpoint, {{
          method: 'POST',
          headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
          body: formBody()
        }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Request failed');
        pollJob(payload.job_id);
      }} catch (error) {{
        setBusy(false);
        setState('Failed');
        log.textContent = error.message;
      }}
    }}

    async function pollJob(jobId) {{
      const timer = setInterval(async () => {{
        try {{
          const response = await fetch('/api/jobs/' + jobId);
          const job = await response.json();
          setState(job.status.charAt(0).toUpperCase() + job.status.slice(1));
          log.textContent = (job.logs || []).join('\\n') || 'Running...';
          renderMetrics(job.result);
          renderArtifacts(job.result);
          if (job.status === 'completed' || job.status === 'failed') {{
            clearInterval(timer);
            setBusy(false);
            if (job.status === 'failed') {{
              log.textContent = ((job.logs || []).join('\\n') + '\\n' + (job.error || '')).trim();
            }}
          }}
        }} catch (error) {{
          clearInterval(timer);
          setBusy(false);
          setState('Failed');
          log.textContent = error.message;
        }}
      }}, 1200);
    }}

    profileButton.addEventListener('click', () => startJob('/api/profile'));
    synthButton.addEventListener('click', () => startJob('/api/synthesize'));
  </script>
</body>
</html>"""


class AppHandler(BaseHTTPRequestHandler):
    server_version = "SyntheticCADLocal/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(_html_page())
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                payload = job.to_dict() if job else {"error": "Job not found"}
            self._send_json(payload, HTTPStatus.OK if job else HTTPStatus.NOT_FOUND)
            return
        if parsed.path.startswith("/artifacts/"):
            self._send_artifact(parsed.path.removeprefix("/artifacts/"))
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        form = parse_qs(body)
        params = {key: _form_value(form, key) for key in form}
        try:
            if parsed.path == "/api/profile":
                self._require(params, ["csv_path", "profile_out_dir"])
                job_id = _start_job("profile", params, _profile_job)
                self._send_json({"job_id": job_id})
                return
            if parsed.path == "/api/synthesize":
                self._require(params, ["csv_path", "mapping_path", "synthesis_out_dir"])
                job_id = _start_job("synthesize", params, _synthesize_job)
                self._send_json({"job_id": job_id})
                return
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _require(self, params: dict[str, str], names: list[str]) -> None:
        missing = [name for name in names if not params.get(name, "").strip()]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

    def _send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_artifact(self, raw_path: str) -> None:
        rel = Path(unquote(raw_path))
        target = (PROJECT_ROOT / rel).resolve()
        try:
            target.relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            self._send_json({"error": "Artifact path is outside the project"}, HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self._send_json({"error": "Artifact not found"}, HTTPStatus.NOT_FOUND)
            return

        content_types = {
            ".html": "text/html; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".csv": "text/csv; charset=utf-8",
            ".txt": "text/plain; charset=utf-8",
        }
        content_type = content_types.get(target.suffix.lower(), "application/octet-stream")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(target.stat().st_size))
        if target.suffix.lower() in {".csv", ".json", ".txt"}:
            self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
        self.end_headers()
        with target.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                self.wfile.write(chunk)


def serve(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"SyntheticCAD local app running at http://{host}:{port}")
    server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SyntheticCAD local prototype app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
