# SyntheticCAD

SyntheticCAD is a local-first Windows prototype for turning sensitive public
safety or service datasets into synthetic research data. It helps an agency
answer two separate questions:

1. **Utility:** Does the synthetic output preserve useful statistical patterns?
2. **Privacy evidence:** Did source identifiers, exact rows, or rare combinations
   survive into the output?

The application runs on the user's computer. The current pipeline does not send
source data to Google, Modal, SDV services, or another cloud endpoint.

The implementation decisions and mentor-feedback checklist are documented in
[`docs/MVP_METHOD_AND_MENTOR_FEEDBACK.md`](docs/MVP_METHOD_AND_MENTOR_FEEDBACK.md).

## Current MVP

The guided local application provides:

- CSV selection with a native Windows file picker
- Local field profiling and reviewable sensitive-field classifications
- Selective attribute modeling
- Direct-identifier exclusion before model fitting
- Synthetic identifier replacement after sampling
- Rare-category grouping before fitting
- SDV single-table synthesis
- Automatic repair of learned deterministic relationships, such as diagnosis
  code/description mappings and admission date + length of stay = discharge date
- Calibrated local runtime estimates
- A Basic Overview and Advanced Evidence dashboard
- Synthetic CSV, SDV metadata, validation JSON, and sharing disclaimer exports

The primary user-facing methods are:

- **SDV Gaussian Copula:** recommended local default; fast, classical statistical
  modeling
- **SDV CTGAN:** advanced neural option; substantially slower and dependent on
  training epochs and available hardware

The older baseline and empirical pattern-matching prototypes are not exposed in
the Windows workflow. The existing CAD-specific `conditional` and relational
`sdv` CLI paths remain available for engineering work on event/unit datasets.

## Repository Layout

```text
syntheticcad/           Python package and local application
tests/                  Privacy, constraint, runtime, and dashboard tests
configs/mappings/       Reusable CAD mappings
datasets/               Local dataset staging; data is ignored by Git
outputs/                Generated runs; ignored by Git
docs/                   Public-safe static dashboard demo
SyntheticCAD.spec       PyInstaller Windows package definition
```

## Dataset Storage

Raw datasets are not stored in GitHub. Team members can use the shared project
[OneDrive folder](https://1drv.ms/f/c/82fa650f732dbbc5/IgCTnXLhMmZ-R6fY1d6udl-_AX1Kdjvg1DGSyYg052eYfA0?e=CpC7zX)
or any other approved storage location, then select the downloaded CSV from the
application. The file does not need to be copied into a particular repository
subfolder.

## Setup On Windows

From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the local application:

```powershell
python -m syntheticcad.web_app
```

SyntheticCAD opens the local interface in the default browser. It starts at
`http://127.0.0.1:8765/` and automatically selects the next available local port
when that port is already in use.

## Guided Workflow

1. Choose a CSV and profile its fields.
2. Review the suggested roles: direct identifier, quasi-identifier, sensitive
   attribute, record identifier, or model attribute.
3. Select only the fields needed for the research use case.
4. Review the modeled-field count, output row count, method, and runtime range.
5. Generate locally and open the Basic Overview.
6. Inspect privacy evidence, distributions, random samples, and Advanced Evidence
   before downloading or sharing the synthetic CSV.

Direct identifiers can remain selected for the output, but they are excluded
from model fitting and regenerated as explicit synthetic aliases. The dashboard
never embeds direct source identifier values.

## Command Line

Profile sensitive fields:

```powershell
python -m syntheticcad.cli sensitive-profile C:\path\data.csv --out outputs\profile.json
```

Run the recommended SDV single-table pipeline on all fields:

```powershell
python -m syntheticcad.cli synthesize-table C:\path\data.csv --out-dir outputs\sdv_run
```

Model selected fields only:

```powershell
python -m syntheticcad.cli synthesize-table C:\path\data.csv `
  --columns "age,race,sex,offense,offense_date" `
  --method gaussian_copula `
  --out-dir outputs\selected_run
```

The CTGAN option is:

```powershell
python -m syntheticcad.cli synthesize-table C:\path\data.csv `
  --method ctgan --ctgan-epochs 100 --out-dir outputs\ctgan_run
```

## Measured Local Runs

Measurements below were recorded on the current development computer on
2026-07-24. They are calibration points, not guarantees.

| Dataset/run | Source rows | Modeled fields | Training | Total pipeline | SDV quality |
|---|---:|---:|---:|---:|---:|
| Fictitious victim, Gaussian Copula | 20,000 | 5 | 7.1 sec | 30.0 sec | 0.988 |
| Victim three-seed stability check | 20,000 | 5 | 3 seeds | 56.0 sec | 0.9875-0.9881 |
| Fictitious hospital, Gaussian Copula | 49,981 | 12 | 24.1 sec | 56.3 sec | 0.991 |
| Victim CTGAN smoke test | 2,000 | 5 | 5 epochs / 21.4 sec | 34.6 sec | 0.726 |

The five-epoch CTGAN run only verifies that the implementation executes. It is
intentionally undertrained and is not evidence that CTGAN is better than the
Gaussian Copula result. CTGAN runtime grows with rows, selected fields, and
epochs; the application shows a wider estimate before the run.

## Windows Package

Install the packaging dependency and build the on-directory package:

```powershell
python -m pip install -r requirements-package.txt
python -m PyInstaller --noconfirm SyntheticCAD.spec
```

The executable is written to:

```text
dist\SyntheticCAD\SyntheticCAD.exe
```

The packaged app stores generated runs under
`%LOCALAPPDATA%\SyntheticCAD\outputs`. The entire `dist\SyntheticCAD` folder must
be distributed together; the executable is not a standalone one-file build.
The current Windows folder is approximately 0.55 GB because SDV and its CPU
modeling dependencies are bundled. The app opens in a few seconds; the first
synthesis in a new session may spend about half a minute loading those
dependencies before fitting begins.

## Validation Evidence

The dashboard reports:

- SDV Column Shapes and Column Pair Trends
- KS statistics for numeric and date fields
- Total variation distance for categorical fields
- Real and synthetic distributions
- Exact source identifier value and identity-combination overlap
- Exact modeled-row overlap
- Rare-combination exposure
- Distance-to-closest-record benchmark against a real holdout
- Nearest-neighbor distance ratio
- Inferred deterministic constraints and applied repairs
- Runtime by fit, sample, and evaluation stage
- A clear list of supported and unsupported claims

The Basic Overview uses a common gap scale for triage:

- `<= 0.10`: green
- `0.10` to `< 0.50`: review
- `>= 0.50`: high

These colors organize review; they are not a universal acceptance standard.
Field-specific targets should be agreed with the agency and researcher before a
production release.

## Privacy Boundary

This MVP provides empirical privacy screens, not a formal privacy proof. The
Community SDV synthesizers used here do not provide a differential privacy
epsilon. A result with strong fidelity can still carry linkage risk.

The application therefore does **not** claim:

- zero re-identification risk
- formal differential privacy
- automatic readiness to share
- preservation of repeated-person relationships in a single-table model
- protection against every agency-specific linkage field

Closed agency data should remain inside the agency environment. A safer
engagement model is for the agency to run SyntheticCAD locally and share only
the synthetic output or validation report after internal privacy review.

## Tests

Run:

```powershell
python -m unittest discover -v
```

The current tests cover field-role precedence, identifier replacement, datetime
protection during rare-value grouping, inferred relationship repair, runtime
scaling, and exclusion of identifier values from generated dashboards.
