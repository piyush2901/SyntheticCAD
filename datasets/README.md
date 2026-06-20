# Datasets

Raw CAD datasets are not committed to GitHub. The team should use the shared
OneDrive folder as the source for dataset files, then download or sync the CSV
locally before running the scripts.

## How To Use The Shared OneDrive Folder

The Python CLI reads files from the local computer. It does not read directly
from an online OneDrive browser link.

Each team member should do one of the following:

1. Download the CSV from the shared OneDrive folder and place it in `datasets/`.
2. Sync the OneDrive folder locally and pass the synced file path to the CLI.

The repo only documents the expected file name and commands. The actual dataset
file stays in OneDrive.

## Seattle Public CAD Dataset

Current prototype dataset:

- File name: `Call_Data_20260619.csv`
- Example local path: `datasets/Call_Data_20260619.csv`
- Source system: Seattle public CAD call data export
- Usage: public sample data for pipeline testing and validation dashboard work

After downloading the file from OneDrive into `datasets/`, verify the row/column
count by running:

```powershell
python -m syntheticcad.cli profile datasets\Call_Data_20260619.csv --out-dir outputs\seattle_2025_profile
```

If the file is synced somewhere else on your machine, pass that path instead:

```powershell
python -m syntheticcad.cli profile "C:\path\to\Call_Data_20260619.csv" --out-dir outputs\seattle_2025_profile
```

The full dataset used during development had 572,296 rows and 47 columns.
