# SyntheticCAD

SyntheticCAD is an offline, Windows-oriented prototype for turning real CAD
calls-for-service CSV exports into shareable synthetic datasets. The goal is to
help agencies share realistic research data without exposing real people, real
incidents, or address-level records.

This repository currently contains the core data pipeline:

- CSV profiling and field-mapping guidance
- Event/unit grain diagnostics
- Baseline synthetic CAD generation
- Executive validation dashboard
- Researcher-oriented validation JSON
- Synthetic CSV export with disclaimer language

The current implementation is a first engineering spine. It uses `pandas` and
`numpy` so the workflow is easy to run locally while the product direction is
still being refined.

## Repository Layout

```text
syntheticcad/           Core Python package
configs/mappings/       Reusable field mappings for known public datasets
datasets/               Local data staging folder; raw data is not committed
outputs/                Generated reports and synthetic exports; not committed
work/                   Scratch space for generated sample files
requirements.txt        Python dependencies
```

## Dataset Storage

Raw datasets are intentionally excluded from GitHub. Team members should download
the shared dataset from the project cloud storage location and place it locally.

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

## Quick Start With Generated Sample Data

Create a small CAD-like sample file:

```powershell
python -m syntheticcad.cli make-sample --out work\sample_cad.csv
```

Profile the sample file and create a mapping guide:

```powershell
python -m syntheticcad.cli profile work\sample_cad.csv --out-dir outputs\profile
```

Generate a baseline synthetic export:

```powershell
python -m syntheticcad.cli synthesize work\sample_cad.csv --mapping outputs\profile\mapping_suggested.json --out-dir outputs\synthetic_run
```

Generated outputs include:

- `synthetic_cad.csv`
- `validation_report.json`
- `executive_dashboard.html`
- `disclaimer.txt`

## Run The Seattle Public CAD Example

Place the Seattle CSV at any local path. The examples below assume:

```text
datasets/Call_Data_20260619.csv
```

Profile the full file:

```powershell
python -m syntheticcad.cli profile datasets\Call_Data_20260619.csv --out-dir outputs\seattle_2025_profile
```

Generate the synthetic dataset using the checked-in Seattle mapping:

```powershell
python -m syntheticcad.cli synthesize datasets\Call_Data_20260619.csv --mapping configs\mappings\seattle_2025_mvp.json --out-dir outputs\seattle_2025_full_run
```

Open the executive dashboard:

```text
outputs/seattle_2025_full_run/executive_dashboard.html
```

## Current Boundaries

This is not yet a final privacy certification workflow. The baseline generator
avoids direct copying of event IDs, unit IDs, and address strings, but the final
product should use a stronger synthetic data methodology and add privacy risk
tests before any real agency data is shared.

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
