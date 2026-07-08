# SyntheticCAD

SyntheticCAD is an offline, Windows-oriented prototype for turning real CAD
calls-for-service CSV exports into shareable synthetic datasets. The goal is to
help agencies share realistic research data without exposing real people, real
incidents, or address-level records.

This repository currently contains the core data pipeline:

- CSV profiling and field-mapping guidance
- Event/unit grain diagnostics
- SDV-based synthetic CAD generation
- Fast empirical pattern-matching synthesis for large local datasets
- Baseline fallback for engineering smoke tests
- Executive validation dashboard
- Researcher-oriented validation JSON
- Synthetic CSV export with disclaimer language

The current implementation uses `pandas`, `numpy`, and SDV. SDV remains the
primary library-backed synthesis path because it supports multi-table relational
synthesis for event/unit CAD structures. The repo also includes a faster
`pattern` method for large prototype runs where the immediate goal is matching
operational statistics.

## Repository Layout

```text
syntheticcad/           Core Python package
configs/mappings/       Reusable field mappings for known public datasets
datasets/               Local data staging folder; raw data is not committed
outputs/                Generated reports and synthetic exports; not committed
requirements.txt        Python dependencies
```

## Dataset Storage

Raw datasets are intentionally excluded from GitHub. Team members should download
the shared dataset from the project cloud storage [OneDrive link](https://1drv.ms/f/c/82fa650f732dbbc5/IgCTnXLhMmZ-R6fY1d6udl-_AX1Kdjvg1DGSyYg052eYfA0?e=CpC7zX).

Example local path for the Seattle sample dataset:

```text
datasets/Call_Data_20260619.csv
```

The same file can also be referenced from any other local path when running the
CLI. For details, see `datasets/README.md`.

## Setup On Windows

From a PowerShell terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.

## Run The Seattle Public CAD Example

Place the Seattle CSV at any local path. The examples below assume:

```text
datasets/Call_Data_20260619.csv
```

Profile the full file:

```powershell
python -m syntheticcad.cli profile datasets\Call_Data_20260619.csv --out-dir outputs\seattle_2025_profile
```

Generate the synthetic dataset using the checked-in Seattle mapping and the fast
pattern-matching method:

```powershell
python -m syntheticcad.cli synthesize datasets\Call_Data_20260619.csv --mapping configs\mappings\seattle_2025_mvp.json --out-dir outputs\seattle_pattern_full_run --method pattern
```

The `synthesize` command uses SDV by default. The `pattern` method is faster and
usually matches the visible operational statistics more closely because it uses
event/unit templates, new IDs, timestamp randomization, paired coordinate jitter,
and address-like location replacement. It is useful for full-file MVP demos, but
it should be documented separately from SDV because it is not a formal privacy
model.

For fast smoke tests with the simplest dependency-light generator, run:

```powershell
python -m syntheticcad.cli synthesize datasets\Call_Data_20260619.csv --mapping configs\mappings\seattle_2025_mvp.json --out-dir outputs\baseline_run --method baseline
```

Open the executive dashboard:

```text
outputs/seattle_pattern_full_run/executive_dashboard.html
```

Generated outputs include:

- `synthetic_cad.csv`
- `validation_report.json`
- `executive_dashboard.html`
- `disclaimer.txt`

## Public Demo Dashboard

This repository includes a static GitHub Pages demo in `docs/index.html`. It is
generated from the public Seattle sample workflow and is intended only as a
shareable visual demonstration of the validation dashboard.

To publish it from GitHub:

1. Open the repository on GitHub.
2. Go to **Settings** > **Pages**.
3. Set **Source** to **Deploy from a branch**.
4. Select the `main` branch and the `/docs` folder.
5. Save the settings.

Do not place raw datasets, closed agency exports, or large synthetic CSV files in
`docs/`. The live demo should contain only static, public-safe artifacts.

## Run The Local Windows Prototype

Start the local browser app from PowerShell:

```powershell
python -m syntheticcad.web_app --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765/
```

The prototype runs entirely on the local machine. It accepts a CSV file path,
mapping JSON path, synthesis method, row limit, seed, and output folder. Use
`pattern` for the fastest full-file MVP run, `sdv` for library-backed synthesis
checks, and `baseline` for the simplest smoke tests.

## Current Boundaries

This is not yet a final privacy certification workflow. SDV is the strongest
library-backed path for the PRD, the `pattern` method is the fastest current
full-file demo path, and the baseline generator remains available for
engineering smoke tests. The final product should still add privacy risk tests
before any real agency data is shared.

The Seattle mapping intentionally uses `Dispatch Neighborhood` instead of
`Dispatch Address` as the location field. Address-level fields should generally
be excluded from synthetic exports unless there is a clear privacy-reviewed need.

## Mentor Review Targets

The most useful early review areas are:

- Does the field mapping reflect real CAD export terminology?
- Does the event/unit diagnostic catch the one-to-many structure correctly?
- Does the executive dashboard answer what an agency leader needs to know?
- Which fields should always be excluded from export by default?
- What validation evidence would make the project credible to researchers and
  agency legal/compliance teams?
