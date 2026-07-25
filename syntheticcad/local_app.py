"""Guided local browser application for SyntheticCAD."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
import uuid
import webbrowser

from syntheticcad.disclaimer import REQUIRED_DISCLAIMER
from syntheticcad.profiling import read_csv, write_json
from syntheticcad.sensitive import field_profile
from syntheticcad.tabular import (
    SUPPORTED_METHODS,
    estimate_runtime,
    synthesize_single_table,
)
from syntheticcad.tabular_dashboard import write_tabular_dashboard


if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(
        os.environ.get("LOCALAPPDATA", str(Path.home()))
    ) / "SyntheticCAD"
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_PORT = 8765
JOBS: dict[str, "JobState"] = {}
JOBS_LOCK = threading.Lock()
PROFILE_CACHE: dict[str, dict[str, Any]] = {}


@dataclass
class JobState:
    id: str
    status: str = "queued"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None

    def log(self, message: str) -> None:
        with JOBS_LOCK:
            self.logs.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
            )

    def to_dict(self) -> dict[str, Any]:
        end = self.finished_at or time.time()
        return {
            "id": self.id,
            "status": self.status,
            "elapsed_seconds": round(end - self.started_at, 1),
            "logs": list(self.logs),
            "result": self.result,
            "error": self.error,
        }


def _resolve_path(value: str, must_exist: bool = False) -> Path:
    path = Path(value.strip().strip('"')).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def _artifact(path: Path, label: str, primary: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    relative = resolved.relative_to(PROJECT_ROOT.resolve())
    return {
        "label": label,
        "path": str(resolved),
        "url": "/artifacts/" + quote(str(relative).replace("\\", "/"), safe="/"),
        "primary": primary,
    }


def _pick_csv() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:  # pragma: no cover - Windows runtime boundary
        raise RuntimeError("The Windows file picker is unavailable.") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilename(
            title="Choose a CSV dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return selected


def _profile_csv(path: Path) -> dict[str, Any]:
    cache_key = f"{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
    if cache_key in PROFILE_CACHE:
        return PROFILE_CACHE[cache_key]
    data = read_csv(path)
    profile = field_profile(data)
    profile.update(
        {
            "file_name": path.name,
            "file_path": str(path),
            "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
            "default_selected": [column for column in data.columns],
        }
    )
    PROFILE_CACHE.clear()
    PROFILE_CACHE[cache_key] = profile
    return profile


def _run_job(params: dict[str, Any], job: JobState) -> dict[str, Any]:
    csv_path = _resolve_path(str(params["csv_path"]), must_exist=True)
    selected = [
        str(column)
        for column in params.get("selected_columns", [])
        if str(column).strip()
    ]
    method = str(params.get("method", "gaussian_copula"))
    rows = int(params.get("rows") or 0) or None
    seed = int(params.get("seed") or 42)
    rare_threshold = max(2, int(params.get("rare_threshold") or 5))
    epochs = max(1, int(params.get("ctgan_epochs") or 100))
    repeat_runs = 3 if int(params.get("repeat_runs") or 1) == 3 else 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / f"{csv_path.stem}_sdv_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    job.log(f"Reading {csv_path.name}")
    source = read_csv(csv_path, usecols=selected)
    job.log(f"Loaded {len(source):,} rows and {len(selected):,} selected fields")
    job.log(f"Fitting {SUPPORTED_METHODS[method]['label']} locally")
    result = synthesize_single_table(
        source,
        selected_columns=selected,
        rows=rows,
        method=method,
        seed=seed,
        rare_threshold=rare_threshold,
        ctgan_epochs=epochs,
    )
    consistency_runs = [
        {
            "seed": seed,
            "quality_score": result.report["quality"]["overall_score"],
            "runtime_seconds": result.report["runtime"]["total_seconds"],
            "field_gaps": {
                item["column"]: item.get("gap")
                for item in result.report["quality"]["column_metrics"]
            },
        }
    ]
    for run_index in range(1, repeat_runs):
        run_seed = seed + run_index
        job.log(
            f"Running stability check {run_index + 1} of {repeat_runs} "
            f"with seed {run_seed}"
        )
        repeated = synthesize_single_table(
            source,
            selected_columns=selected,
            rows=rows,
            method=method,
            seed=run_seed,
            rare_threshold=rare_threshold,
            ctgan_epochs=epochs,
        )
        consistency_runs.append(
            {
                "seed": run_seed,
                "quality_score": repeated.report["quality"]["overall_score"],
                "runtime_seconds": repeated.report["runtime"]["total_seconds"],
                "field_gaps": {
                    item["column"]: item.get("gap")
                    for item in repeated.report["quality"]["column_metrics"]
                },
            }
        )
    quality_scores = [item["quality_score"] for item in consistency_runs]
    all_fields = sorted(
        {
            column
            for item in consistency_runs
            for column in item["field_gaps"]
        }
    )
    result.report["consistency"] = {
        "run_count": repeat_runs,
        "runs": consistency_runs,
        "total_runtime_seconds": round(
            sum(item["runtime_seconds"] for item in consistency_runs),
            3,
        ),
        "quality_score_mean": round(sum(quality_scores) / len(quality_scores), 4),
        "quality_score_min": round(min(quality_scores), 4),
        "quality_score_max": round(max(quality_scores), 4),
        "field_gap_spread": {
            column: round(
                max(
                    float(item["field_gaps"][column])
                    for item in consistency_runs
                    if item["field_gaps"].get(column) is not None
                )
                - min(
                    float(item["field_gaps"][column])
                    for item in consistency_runs
                    if item["field_gaps"].get(column) is not None
                ),
                4,
            )
            for column in all_fields
            if any(
                item["field_gaps"].get(column) is not None
                for item in consistency_runs
            )
        },
    }
    result.report["runtime"]["total_all_runs_seconds"] = result.report[
        "consistency"
    ]["total_runtime_seconds"]
    job.log(
        f"Generated {len(result.dataframe):,} rows in "
        f"{result.report['runtime']['total_all_runs_seconds']:.1f} seconds "
        f"across {repeat_runs} run{'s' if repeat_runs != 1 else ''}"
    )

    csv_output = out_dir / "synthetic_data.csv"
    report_output = out_dir / "validation_report.json"
    metadata_output = out_dir / "sdv_metadata.json"
    disclaimer_output = out_dir / "disclaimer.txt"
    dashboard_output = out_dir / "validation_dashboard.html"
    result.dataframe.to_csv(csv_output, index=False)
    write_json(result.report, report_output)
    write_json(result.report["metadata"], metadata_output)
    disclaimer_output.write_text(REQUIRED_DISCLAIMER + "\n", encoding="utf-8")
    write_tabular_dashboard(
        result.model_data,
        result.dataframe,
        result.report,
        dashboard_output,
    )
    job.log("Wrote the dashboard, synthetic CSV, and evidence files")

    privacy = result.report["privacy"]
    quality = result.report["quality"]
    exact_identity = privacy["direct_identifier_overlap"][
        "exact_identity_combination"
    ]
    return {
        "summary": {
            "real_rows": len(source),
            "synthetic_rows": len(result.dataframe),
            "modeled_fields": len(result.report["pipeline"]["modeled_columns"]),
            "replaced_identifiers": len(
                result.report["pipeline"]["excluded_identifier_columns"]
            ),
            "quality_score": quality["overall_score"],
            "identity_matches": exact_identity["matching_synthetic_rows"],
            "runtime_seconds": result.report["runtime"]["total_all_runs_seconds"],
            "consistency_runs": repeat_runs,
            "output_folder": str(out_dir),
        },
        "artifacts": [
            _artifact(dashboard_output, "Open Validation Dashboard", primary=True),
            _artifact(csv_output, "Synthetic CSV"),
            _artifact(report_output, "Validation Report"),
            _artifact(metadata_output, "SDV Metadata"),
            _artifact(disclaimer_output, "Sharing Disclaimer"),
        ],
    }


def _start_job(params: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = JobState(id=job_id)
    with JOBS_LOCK:
        JOBS[job_id] = job

    def run() -> None:
        with JOBS_LOCK:
            job.status = "running"
        try:
            job.result = _run_job(params, job)
            with JOBS_LOCK:
                job.status = "completed"
                job.finished_at = time.time()
        except Exception as exc:  # pragma: no cover - app boundary
            job.log(traceback.format_exc().strip())
            with JOBS_LOCK:
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = time.time()

    threading.Thread(target=run, daemon=True).start()
    return job_id


def _page() -> str:
    methods = "".join(
        f'<option value="{escape(key)}">{escape(value["label"])}</option>'
        for key, value in SUPPORTED_METHODS.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SyntheticCAD</title>
  <style>
    :root {{
      --ink:#18211e; --muted:#66716c; --line:#d6dcd7; --paper:#fff;
      --bg:#f2f4f1; --accent:#245f57; --accent-dark:#19483f;
      --blue:#28657c; --amber:#9a6500; --red:#9b3b33; --green:#206a49;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--bg); font-family:"Segoe UI",Arial,sans-serif; letter-spacing:0; }}
    header {{ height:64px; padding:0 24px; background:#fff; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; gap:18px; }}
    .brand {{ font-weight:800; font-size:20px; }} .brand small {{ color:var(--muted); font-size:12px; font-weight:500; margin-left:9px; }}
    .local {{ color:var(--green); font-size:12px; font-weight:700; }}
    .shell {{ max-width:1360px; margin:0 auto; display:grid; grid-template-columns:220px minmax(0,1fr); min-height:calc(100vh - 64px); }}
    aside {{ border-right:1px solid var(--line); padding:22px 18px; }}
    .step-link {{ display:grid; grid-template-columns:28px 1fr; gap:9px; align-items:center; padding:10px 8px; color:var(--muted); font-size:13px; }}
    .step-link i {{ width:27px; height:27px; border:1px solid #bfc8c2; border-radius:50%; font-style:normal; display:grid; place-items:center; font-size:12px; font-weight:800; }}
    .step-link.active {{ color:var(--ink); font-weight:750; }} .step-link.active i {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
    .step-link.done i {{ background:#e3f1e9; color:var(--green); border-color:#a9cbb9; }}
    main {{ padding:24px 28px 42px; min-width:0; }}
    .step {{ display:none; }} .step.active {{ display:block; }}
    h1 {{ font-size:24px; margin:0; }} .lead {{ color:var(--muted); margin:6px 0 22px; font-size:14px; max-width:760px; line-height:1.5; }}
    .panel {{ background:#fff; border:1px solid var(--line); padding:18px; margin-bottom:14px; }}
    .panel-head {{ display:flex; justify-content:space-between; gap:18px; align-items:flex-start; margin-bottom:14px; }}
    h2 {{ font-size:15px; margin:0; }} .panel-head p {{ margin:4px 0 0; color:var(--muted); font-size:12px; }}
    button {{ min-height:38px; border:0; border-radius:5px; padding:8px 13px; font:inherit; font-size:13px; font-weight:750; cursor:pointer; }}
    button.primary {{ background:var(--accent); color:#fff; }} button.primary:hover {{ background:var(--accent-dark); }}
    button.secondary {{ background:#fff; color:var(--ink); border:1px solid #bcc6c0; }} button:disabled {{ opacity:.5; cursor:not-allowed; }}
    input,select {{ min-height:38px; border:1px solid #bdc6c0; border-radius:5px; padding:8px 10px; font:inherit; font-size:13px; background:#fff; color:var(--ink); }}
    input:focus,select:focus {{ outline:3px solid rgba(36,95,87,.14); border-color:var(--accent); }}
    .file-row {{ display:grid; grid-template-columns:auto minmax(0,1fr) auto; gap:10px; align-items:center; }}
    #csv-path {{ width:100%; }} .file-facts {{ display:grid; grid-template-columns:repeat(4,1fr); border:1px solid var(--line); margin-top:14px; }}
    .fact {{ padding:11px 13px; border-right:1px solid var(--line); }} .fact:last-child {{ border-right:0; }}
    .fact small {{ color:var(--muted); display:block; font-size:10px; text-transform:uppercase; font-weight:750; }} .fact b {{ font-size:17px; display:block; margin-top:5px; }}
    .actions {{ display:flex; justify-content:flex-end; gap:9px; margin-top:16px; }}
    .field-toolbar {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
    .field-toolbar input {{ min-width:230px; }} .selection-count {{ color:var(--muted); font-size:12px; margin-left:auto; }}
    .field-table-wrap {{ border:1px solid var(--line); max-height:520px; overflow:auto; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; background:#fff; }}
    th,td {{ padding:9px 10px; border-bottom:1px solid #e4e8e5; text-align:left; white-space:nowrap; }}
    th {{ background:#edf1ee; position:sticky; top:0; z-index:1; font-size:11px; }} td:first-child,th:first-child {{ width:40px; text-align:center; }}
    .role {{ padding:3px 7px; border-radius:3px; font-size:10px; font-weight:800; text-transform:uppercase; }}
    .role.direct_identifier,.role.record_identifier {{ background:#f6e2dd; color:var(--red); }}
    .role.quasi_identifier {{ background:#fff0c9; color:var(--amber); }}
    .role.sensitive_attribute {{ background:#e4edf5; color:#315d79; }}
    .role.model_attribute {{ background:#e4f0e8; color:var(--green); }}
    .config-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    label {{ display:block; font-size:12px; font-weight:750; margin-bottom:6px; }} .control input,.control select {{ width:100%; }}
    .estimate {{ border-left:4px solid var(--blue); background:#eef4f7; padding:13px 14px; margin-top:14px; display:flex; justify-content:space-between; gap:15px; align-items:center; }}
    .estimate b {{ font-size:18px; }} .estimate span {{ color:var(--muted); font-size:11px; max-width:600px; }}
    details {{ margin-top:14px; border-top:1px solid var(--line); padding-top:11px; }} summary {{ cursor:pointer; font-size:12px; font-weight:750; }}
    .run-panel {{ display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:14px; }}
    .progress {{ height:8px; background:#e5e9e6; overflow:hidden; margin:12px 0; }} .progress i {{ height:100%; display:block; width:15%; background:var(--accent); animation:pulse 1.5s infinite alternate; }}
    @keyframes pulse {{ from {{ opacity:.45; }} to {{ opacity:1; }} }}
    pre {{ margin:0; background:#111916; color:#d9ebe4; padding:12px; min-height:220px; max-height:360px; overflow:auto; font-size:11px; line-height:1.5; white-space:pre-wrap; }}
    .result-grid {{ display:grid; grid-template-columns:repeat(5,1fr); border:1px solid var(--line); }}
    .result-links {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin-top:14px; }}
    .artifact {{ border:1px solid var(--line); background:#fff; padding:13px; text-decoration:none; color:var(--ink); display:flex; justify-content:space-between; align-items:center; }}
    .artifact.primary {{ background:var(--accent); color:#fff; grid-column:1/-1; }} .artifact span {{ color:inherit; font-size:12px; font-weight:750; }}
    .notice {{ padding:11px 13px; background:#fff5d9; border-left:4px solid #d19618; font-size:12px; line-height:1.5; }}
    .hidden {{ display:none!important; }}
    @media(max-width:900px) {{ .shell {{ grid-template-columns:1fr; }} aside {{ border-right:0; border-bottom:1px solid var(--line); display:flex; overflow:auto; padding:8px; }} .step-link {{ min-width:145px; }} main {{ padding:18px; }} .config-grid,.run-panel {{ grid-template-columns:1fr; }} .file-facts,.result-grid {{ grid-template-columns:repeat(2,1fr); }} }}
    @media(max-width:580px) {{ header {{ padding:0 14px; }} .brand small {{ display:none; }} .file-row {{ grid-template-columns:1fr; }} .field-toolbar input {{ width:100%; }} .result-links {{ grid-template-columns:1fr; }} .artifact.primary {{ grid-column:auto; }} }}
  </style>
</head>
<body>
  <header><div class="brand">SyntheticCAD <small>Local data workspace</small></div><div class="local">Data stays on this computer</div></header>
  <div class="shell">
    <aside>
      <div class="step-link active" data-step-link="1"><i>1</i><span>Choose data</span></div>
      <div class="step-link" data-step-link="2"><i>2</i><span>Select fields</span></div>
      <div class="step-link" data-step-link="3"><i>3</i><span>Configure run</span></div>
      <div class="step-link" data-step-link="4"><i>4</i><span>Review results</span></div>
    </aside>
    <main>
      <section class="step active" data-step="1">
        <h1>Choose a dataset</h1>
        <p class="lead">Select a CSV from this computer. SyntheticCAD profiles it locally and identifies fields that need privacy treatment.</p>
        <div class="panel">
          <div class="panel-head"><div><h2>Source CSV</h2><p>No file is uploaded to a server or cloud service.</p></div></div>
          <div class="file-row">
            <button type="button" class="secondary" id="pick-file">Choose CSV</button>
            <input id="csv-path" placeholder="Choose a file or paste its local path">
            <button type="button" class="primary" id="profile-file">Profile fields</button>
          </div>
          <div id="profile-message" class="notice hidden" style="margin-top:14px"></div>
          <div id="file-facts" class="file-facts hidden"></div>
        </div>
      </section>

      <section class="step" data-step="2">
        <h1>Select fields</h1>
        <p class="lead">Choose what the synthetic dataset should contain. Identifier fields are replaced, not used to fit the model.</p>
        <div class="panel">
          <div class="panel-head">
            <div><h2>Field profile</h2><p>Detected roles are suggestions. Review them before using closed agency data.</p></div>
            <div class="field-toolbar"><input id="field-search" placeholder="Filter fields"><button type="button" class="secondary" id="select-modelable">Modelable only</button></div>
          </div>
          <div class="selection-count" id="selection-count"></div>
          <div class="field-table-wrap"><table><thead><tr><th><input type="checkbox" id="toggle-all" checked></th><th>Field</th><th>Role</th><th>Type</th><th>Unique</th><th>Missing</th><th>Action</th></tr></thead><tbody id="field-rows"></tbody></table></div>
          <div class="actions"><button type="button" class="secondary" data-back="1">Back</button><button type="button" class="primary" id="to-config">Continue</button></div>
        </div>
      </section>

      <section class="step" data-step="3">
        <h1>Configure the run</h1>
        <p class="lead">The recommended method is optimized for a quick local first pass. Advanced neural training is available when the extra runtime is justified.</p>
        <div class="panel">
          <div class="config-grid">
            <div class="control"><label for="method">Synthesis method</label><select id="method">{methods}</select></div>
            <div class="control"><label for="rows">Synthetic rows</label><input id="rows" type="number" min="1"></div>
          </div>
          <div class="estimate"><div><small>Expected local runtime</small><b id="runtime-estimate">-</b></div><span id="runtime-note"></span></div>
          <details>
            <summary>Advanced settings</summary>
            <div class="config-grid" style="margin-top:12px">
              <div class="control"><label for="rare-threshold">Group categories with fewer than</label><input id="rare-threshold" type="number" min="2" value="5"></div>
              <div class="control"><label for="seed">Random seed</label><input id="seed" type="number" value="42"></div>
              <div class="control" id="epochs-control"><label for="epochs">CTGAN epochs</label><input id="epochs" type="number" min="1" value="100"></div>
              <div class="control"><label for="repeat-runs">Stability check</label><select id="repeat-runs"><option value="1">One run</option><option value="3">Three seeds</option></select></div>
            </div>
          </details>
          <div class="notice" style="margin-top:14px">This run will measure fidelity and observed privacy exposure. It will not claim formal differential privacy or zero re-identification risk.</div>
          <div class="actions"><button type="button" class="secondary" data-back="2">Back</button><button type="button" class="primary" id="start-run">Generate synthetic data</button></div>
        </div>
      </section>

      <section class="step" data-step="4">
        <h1 id="result-title">Running locally</h1>
        <p class="lead" id="result-lead">Keep this window open while SyntheticCAD fits the model, generates rows, and evaluates the result.</p>
        <div id="running-view" class="run-panel">
          <div class="panel"><h2 id="job-status">Preparing the run</h2><div class="progress"><i></i></div><pre id="job-log">Waiting for the local job...</pre></div>
          <div class="panel"><h2>What happens next</h2><p class="lead" style="font-size:12px;margin-bottom:0">The result includes a Basic Overview, Advanced Evidence, synthetic CSV, SDV metadata, and a sharing disclaimer.</p></div>
        </div>
        <div id="completed-view" class="hidden">
          <div class="result-grid" id="result-facts"></div>
          <div class="result-links" id="result-links"></div>
          <div class="actions"><button type="button" class="secondary" id="new-run">Start another run</button></div>
        </div>
      </section>
    </main>
  </div>
  <script>
    const state = {{ profile:null, selected:[], jobId:null }};
    const $ = id => document.getElementById(id);
    function showStep(number) {{
      document.querySelectorAll('.step').forEach(step => step.classList.toggle('active', step.dataset.step === String(number)));
      document.querySelectorAll('.step-link').forEach(link => {{
        const value = Number(link.dataset.stepLink); link.classList.toggle('active', value === number); link.classList.toggle('done', value < number);
      }});
    }}
    document.querySelectorAll('[data-back]').forEach(button => button.addEventListener('click', () => showStep(Number(button.dataset.back))));
    async function request(url, payload={{}}) {{
      const response = await fetch(url, {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(payload)}});
      const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Request failed'); return data;
    }}
    $('pick-file').addEventListener('click', async () => {{
      try {{ const data = await request('/api/pick-file'); if (data.path) $('csv-path').value = data.path; }} catch (error) {{ showProfileError(error.message); }}
    }});
    function showProfileError(message) {{ $('profile-message').textContent = message; $('profile-message').classList.remove('hidden'); }}
    $('profile-file').addEventListener('click', async () => {{
      const button = $('profile-file'); button.disabled = true; button.textContent = 'Profiling...'; $('profile-message').classList.add('hidden');
      try {{
        state.profile = await request('/api/profile', {{csv_path:$('csv-path').value}});
        state.selected = [...state.profile.default_selected]; $('rows').value = state.profile.rows;
        $('file-facts').innerHTML = [
          ['File', state.profile.file_name], ['Rows', state.profile.rows.toLocaleString()], ['Columns', state.profile.columns.toLocaleString()], ['Size', `${{state.profile.file_size_mb}} MB`]
        ].map(([label,value]) => `<div class="fact"><small>${{label}}</small><b>${{value}}</b></div>`).join('');
        $('file-facts').classList.remove('hidden'); renderFields(); showStep(2);
      }} catch (error) {{ showProfileError(error.message); }}
      finally {{ button.disabled = false; button.textContent = 'Profile fields'; }}
    }});
    function roleLabel(role) {{ return role.replaceAll('_',' '); }}
    function renderFields() {{
      const query = $('field-search').value.toLowerCase();
      $('field-rows').innerHTML = state.profile.fields.filter(item => item.column.toLowerCase().includes(query)).map(item => `
        <tr><td><input class="field-check" type="checkbox" value="${{escapeAttr(item.column)}}" ${{state.selected.includes(item.column)?'checked':''}}></td>
        <td><b>${{escapeHtml(item.column)}}</b></td><td><span class="role ${{item.role}}">${{roleLabel(item.role)}}</span></td><td>${{item.sdtype}}</td>
        <td>${{item.unique_values.toLocaleString()}}</td><td>${{item.missing_pct}}%</td><td>${{escapeHtml(item.recommended_action)}}</td></tr>`).join('');
      document.querySelectorAll('.field-check').forEach(check => check.addEventListener('change', () => {{
        if (check.checked && !state.selected.includes(check.value)) state.selected.push(check.value);
        if (!check.checked) state.selected = state.selected.filter(value => value !== check.value);
        updateSelection();
      }})); updateSelection();
    }}
    function updateSelection() {{
      const roles = Object.fromEntries(state.profile.fields.map(item => [item.column,item.role]));
      const modeled = state.selected.filter(column => !['direct_identifier','record_identifier'].includes(roles[column])).length;
      const replaced = state.selected.length - modeled;
      $('selection-count').textContent = `${{state.selected.length}} selected | ${{modeled}} modeled | ${{replaced}} replaced`;
      updateEstimate();
    }}
    $('field-search').addEventListener('input', renderFields);
    $('toggle-all').addEventListener('change', event => {{ state.selected = event.target.checked ? [...state.profile.default_selected] : []; renderFields(); }});
    $('select-modelable').addEventListener('click', () => {{ state.selected = state.profile.fields.filter(item => !['direct_identifier','record_identifier'].includes(item.role)).map(item => item.column); renderFields(); }});
    $('to-config').addEventListener('click', () => {{
      if (!state.selected.length) return alert('Select at least one field.');
      const roles = Object.fromEntries(state.profile.fields.map(item => [item.column,item.role]));
      if (!state.selected.some(column => !['direct_identifier','record_identifier'].includes(roles[column]))) return alert('Select at least one modelable field.');
      updateEstimate(); showStep(3);
    }});
    function updateEstimate() {{
      if (!state.profile) return;
      const roles = Object.fromEntries(state.profile.fields.map(item => [item.column,item.role]));
      const columns = state.selected.filter(column => !['direct_identifier','record_identifier'].includes(roles[column])).length;
      const rows = Number($('rows').value || state.profile.rows); const method = $('method').value; const epochs = Number($('epochs').value || 100); const repeats = Number($('repeat-runs').value || 1);
      const cells = Math.max(rows,1)*Math.max(columns,1); let center,basis;
      if (method === 'ctgan') {{ center=15+cells*Math.max(epochs,1)*.000128; basis='Estimate is calibrated from a local 2,000-row CTGAN smoke run; epochs and hardware matter.'; }}
      else {{ center=12+cells*.000032+columns*columns*.03; basis='Estimate is calibrated from the full victim and hospital Gaussian Copula runs on this machine.'; }}
      const label = seconds => seconds<60?`${{Math.ceil(seconds)}} sec`:seconds<3600?`${{Math.ceil(seconds/60)}} min`:`${{(seconds/3600).toFixed(1)}} hr`;
      center *= repeats;
      $('runtime-estimate').textContent = `${{label(Math.max(2,center*.65))}} - ${{label(Math.max(4,center*1.7))}}`;
      $('runtime-note').textContent = `${{rows.toLocaleString()}} rows x ${{columns}} modeled fields x ${{repeats}} run${{repeats===1?'':'s'}}. ${{basis}} This is a planning estimate, not a promise.`;
      $('epochs-control').classList.toggle('hidden', method !== 'ctgan');
    }}
    ['method','rows','epochs','repeat-runs'].forEach(id => $(id).addEventListener('input', updateEstimate));
    $('start-run').addEventListener('click', async () => {{
      showStep(4); $('completed-view').classList.add('hidden'); $('running-view').classList.remove('hidden');
      try {{
        const data = await request('/api/run', {{csv_path:$('csv-path').value,selected_columns:state.selected,method:$('method').value,rows:Number($('rows').value),seed:Number($('seed').value),rare_threshold:Number($('rare-threshold').value),ctgan_epochs:Number($('epochs').value),repeat_runs:Number($('repeat-runs').value)}});
        state.jobId=data.job_id; pollJob();
      }} catch(error) {{ finishError(error.message); }}
    }});
    async function pollJob() {{
      const response=await fetch(`/api/job?id=${{encodeURIComponent(state.jobId)}}`); const job=await response.json();
      $('job-status').textContent = job.status === 'running' ? `Running | ${{job.elapsed_seconds.toFixed(1)}} sec` : job.status;
      $('job-log').textContent = job.logs.join('\\n') || 'Starting...'; $('job-log').scrollTop=$('job-log').scrollHeight;
      if (job.status === 'completed') return finishSuccess(job.result);
      if (job.status === 'failed') return finishError(job.error);
      setTimeout(pollJob,1000);
    }}
    function finishSuccess(result) {{
      $('result-title').textContent='Run complete'; $('result-lead').textContent='Review the visual evidence before downloading or sharing the synthetic output.';
      $('running-view').classList.add('hidden'); $('completed-view').classList.remove('hidden');
      const s=result.summary; $('result-facts').innerHTML=[
        ['Real rows',s.real_rows.toLocaleString()],['Synthetic rows',s.synthetic_rows.toLocaleString()],['Modeled fields',s.modeled_fields],['Identity matches',s.identity_matches],['SDV quality',s.quality_score.toFixed(3)]
      ].map(([label,value])=>`<div class="fact"><small>${{label}}</small><b>${{value}}</b></div>`).join('');
      $('result-links').innerHTML=result.artifacts.map(item=>`<a class="artifact ${{item.primary?'primary':''}}" href="${{item.url}}" ${{item.primary?'target=\"_blank\"':''}}><span>${{escapeHtml(item.label)}}</span><b>Open</b></a>`).join('');
    }}
    function finishError(message) {{ $('result-title').textContent='Run stopped'; $('result-lead').textContent=message; $('job-status').textContent='Failed'; }}
    $('new-run').addEventListener('click',()=>showStep(1));
    function escapeHtml(value) {{ const div=document.createElement('div'); div.textContent=String(value); return div.innerHTML; }}
    function escapeAttr(value) {{ return escapeHtml(value).replaceAll('\"','&quot;'); }}
    updateEstimate();
  </script>
</body>
</html>"""


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_html(_page())
        if parsed.path == "/api/job":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                payload = job.to_dict() if job else None
            if payload is None:
                return self._send_json({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
            return self._send_json(payload)
        if parsed.path.startswith("/artifacts/"):
            return self._send_artifact(parsed.path[len("/artifacts/") :])
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            parsed = urlparse(self.path)
            if parsed.path == "/api/pick-file":
                return self._send_json({"path": _pick_csv()})
            if parsed.path == "/api/profile":
                path = _resolve_path(str(payload.get("csv_path", "")), must_exist=True)
                return self._send_json(_profile_csv(path))
            if parsed.path == "/api/estimate":
                return self._send_json(
                    estimate_runtime(
                        int(payload.get("rows", 0)),
                        int(payload.get("modeled_columns", 0)),
                        str(payload.get("method", "gaussian_copula")),
                        int(payload.get("epochs", 100)),
                    )
                )
            if parsed.path == "/api/run":
                return self._send_json({"job_id": _start_job(payload)})
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover - app boundary
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_artifact(self, raw_path: str) -> None:
        target = (PROJECT_ROOT / unquote(raw_path)).resolve()
        try:
            target.relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            return self._send_json({"error": "Invalid path"}, HTTPStatus.FORBIDDEN)
        if not target.is_file():
            return self._send_json({"error": "File not found"}, HTTPStatus.NOT_FOUND)
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".csv": "text/csv; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".txt": "text/plain; charset=utf-8",
        }
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            content_types.get(target.suffix.lower(), "application/octet-stream"),
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(host: str, port: int, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer((host, port), AppHandler)
    url = f"http://{host}:{port}/"
    print(f"SyntheticCAD is running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _available_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            try:
                candidate.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(
        f"No available local port was found between {preferred} and {preferred + 19}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SyntheticCAD local app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    port = _available_port(args.host, args.port)
    serve(args.host, port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
