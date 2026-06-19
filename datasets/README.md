# Datasets

Raw CAD datasets are not committed to GitHub. Store shared datasets in the
project SharePoint or cloud storage location and copy them into this folder, or
reference them from any local path when running the CLI.

## Shared Storage Option

The SharePoint layout does not have to match the repo folder layout. A simple
single SharePoint folder is enough while the project is small.

If the number of datasets grows, one reasonable option is to use a single
SharePoint root folder and organize files by source and dataset type:

```text
SyntheticCAD Datasets/
  public-cad/
    seattle/
      Call_Data_20260619.csv
    baltimore/
      911_CallsForService_2025.csv
  public-crime/
  private-agency-samples/
```

This is only an example. The important part is that each dataset has a clear
file name, source, date downloaded, and mapping file. See `catalog.example.yml`
for the metadata shape the repo can track.

## Seattle Public CAD Dataset

Current prototype dataset:

- File name: `Call_Data_20260619.csv`
- Example local path: `datasets/Call_Data_20260619.csv`
- Source system: Seattle public CAD call data export
- Usage: public sample data for pipeline testing and validation dashboard work

After downloading the file from shared storage, verify the row/column count by
running:

```powershell
python -m syntheticcad.cli profile datasets\Call_Data_20260619.csv --out-dir outputs\seattle_2025_profile
```

The full dataset used during development had 572,296 rows and 47 columns.
