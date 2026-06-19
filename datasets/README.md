# Datasets

Raw CAD datasets are not committed to GitHub. Store shared datasets in the
project SharePoint or cloud storage location and copy them into this folder, or
reference them from any local path when running the CLI.

## Shared Storage Pattern

Use one SharePoint root folder for the team, then organize datasets under that
root by source and type. The repo should document dataset metadata, not store
the actual files.

Example SharePoint layout:

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

See `catalog.example.yml` for the metadata shape the repo can track.

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

## Team Practice

- Keep raw datasets in cloud storage, not Git.
- Keep cloud-sharing links out of public documentation if they are restricted.
- Record source URL, download date, file name, and checksum when adding a new
  dataset to the shared storage folder.
- Prefer broad geography fields such as neighborhood, precinct, sector, or beat
  over address-level fields for validation and export.
