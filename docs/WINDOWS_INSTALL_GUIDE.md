# Download And Run SyntheticCAD On Windows

SyntheticCAD is a portable Windows application. You do not need Python, and
there is no installer. Download the ZIP file, extract the whole folder, and run
the application from that folder.

## Before You Start

You need:

- Windows 10 or Windows 11
- About 1 GB of free disk space
- A CSV file that you are allowed to use

SyntheticCAD works on your computer. Your CSV is not uploaded to GitHub or a
cloud service.

## 1. Download The App

1. Open the [SyntheticCAD Releases page](https://github.com/piyush2901/SyntheticCAD/releases/latest).
2. Find the **Assets** section.
3. Download `SyntheticCAD-Windows-v0.1.0.zip`.
4. Wait for the download to finish. The file is large because it includes the
   data-modeling software needed to run without Python.

## 2. Extract The ZIP File

Do not run the application from inside the ZIP file.

1. Open your **Downloads** folder.
2. Right-click `SyntheticCAD-Windows-v0.1.0.zip`.
3. Select **Extract All**.
4. Choose a folder and select **Extract**.
5. Open the extracted `SyntheticCAD` folder.

Keep every file in this folder together. Moving only `SyntheticCAD.exe` will
break the application.

## 3. Start SyntheticCAD

1. Double-click `SyntheticCAD.exe`.
2. Wait a few seconds.
3. SyntheticCAD will open in your web browser.

The local address is usually:

```text
http://127.0.0.1:8765/
```

If the browser does not open, enter that address in your browser yourself. The
address works only while SyntheticCAD is running on your computer.

### Windows Security Message

Windows may show **Windows protected your PC** because this prototype is not
digitally signed.

If you downloaded the ZIP from the official SyntheticCAD GitHub release:

1. Select **More info**.
2. Check that the app name is `SyntheticCAD.exe`.
3. Select **Run anyway**.

Do not continue if the file came from another website or an unknown sender.

## 4. Create Synthetic Data

1. Select **Choose CSV**.
2. Choose a CSV file from your computer.
3. Select **Profile fields**.
4. Review the fields and select only the fields needed for your work.
5. Review the suggested privacy role for each field.
6. Choose **SDV Gaussian Copula** for the recommended first run.
7. Review the number of rows and the estimated run time.
8. Start the generation.
9. Open the validation dashboard when the run finishes.

The first generation after starting the app may take extra time while the
modeling software loads.

## 5. Review The Results

Before sharing the synthetic CSV, review:

- the Basic Overview
- the real and synthetic distributions
- exact identifier and row overlap
- rare-combination exposure
- the Advanced Evidence section
- the sharing disclaimer

A good quality score does not prove that a dataset has no privacy risk. An
agency privacy, legal, or data-governance reviewer should approve the output
before it is shared.

## Where Results Are Saved

SyntheticCAD saves generated files here:

```text
%LOCALAPPDATA%\SyntheticCAD\outputs
```

To open that folder:

1. Press `Windows key + R`.
2. Enter `%LOCALAPPDATA%\SyntheticCAD\outputs`.
3. Select **OK**.

Each run has its own folder containing the synthetic CSV, dashboard, validation
report, SDV metadata, and disclaimer.

## Stop The App

Closing the browser tab does not always stop SyntheticCAD.

1. Press `Ctrl + Shift + Esc` to open Task Manager.
2. Find `SyntheticCAD` in the process list.
3. Select it, then select **End task**.

## Remove SyntheticCAD

1. Stop SyntheticCAD.
2. Delete the extracted `SyntheticCAD` application folder.

Generated results are stored separately. Delete
`%LOCALAPPDATA%\SyntheticCAD` only if you also want to remove every generated
CSV and report.

## Common Problems

### The Site Cannot Be Reached

- Make sure `SyntheticCAD.exe` is still running in Task Manager.
- Start `SyntheticCAD.exe` again.
- Try `http://127.0.0.1:8765/` in your browser.
- If another program uses that address, SyntheticCAD will try the next port,
  such as `http://127.0.0.1:8766/`.

### The App Does Not Start

- Make sure you extracted the entire ZIP.
- Keep all extracted files in the same folder.
- Check whether Windows Security or your organization's security software
  blocked the app.
- Ask your IT team before changing organization-managed security settings.

### Generation Takes A Long Time

- Start with SDV Gaussian Copula.
- Select only the fields needed for the analysis.
- Run one seed first.
- CTGAN is an advanced option and can take much longer.

## For Developers

Developers who want to run from source or build a new Windows package should
follow the setup and packaging instructions in the
[main README](../README.md).
