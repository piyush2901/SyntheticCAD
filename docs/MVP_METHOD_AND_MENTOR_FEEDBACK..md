# MVP Method And Mentor Feedback

## Product Position

SyntheticCAD is designed to be run inside the data owner's environment. The
agency keeps the source CSV and can share the synthetic output or validation
report after its own review. The MVP does not require the development team to
receive closed agency data.

The Google Sensitive Data Protection documentation is used as a design
reference for the separation between detection and transformation. SyntheticCAD
does not call the Google API:

https://docs.cloud.google.com/sensitive-data-protection/docs/deidentify-sensitive-data

## Current Pipeline

1. Read the CSV locally.
2. Classify fields as direct identifiers, record identifiers,
   quasi-identifiers, sensitive attributes, or model attributes.
3. Let the user select the fields needed for the intended analysis.
4. Exclude direct and record identifiers before model fitting.
5. Group categorical values that appear fewer than `k=5` times by default.
6. Fit an official SDV single-table synthesizer:
   - Gaussian Copula by default
   - CTGAN as an advanced, slower option
7. Generate the requested number of synthetic rows.
   The selected seed is applied after fitting so that the same seed is
   reproducible and different seeds produce different samples.
8. Regenerate identifier columns as explicit synthetic aliases.
9. Detect and repair deterministic source relationships such as:
   - diagnosis code -> diagnosis description
   - admission date + length of stay = discharge date
10. Measure fidelity, exact overlap, and rare-combination exposure.
11. Write the synthetic CSV, metadata, validation report, dashboard, and
    disclaimer locally.

SDV single-table documentation:

https://docs.sdv.dev/SDV/single-table-data/modeling/synthesizers

## Mentor Feedback Tracking

| Feedback | MVP response |
|---|---|
| Can the tool run multiple times? | Yes. The app supports one run or a three-seed stability check and reports the quality range and largest field-gap spread. |
| Show individual elements, not only statistics | Basic Overview includes randomly selected real and synthetic samples. Direct identifiers are excluded from the embedded real sample. A five-minute window is used when the source has time-of-day values; date-only data uses a one-day window. |
| Avoid terms such as "very close" | The new dashboard shows the measured KS statistic, total variation distance, SDV score, real value, and synthetic value. |
| Add inline distributions | Basic Overview includes an interactive field distribution explorer. |
| Add drill-down and interactivity | Users can select a field, cycle through samples, switch Basic/Advanced views, and sort the technical table. |
| Sort by difference | Every Advanced Evidence table heading is sortable, including gap. |
| Color-code gaps | `<=0.10` is green, `0.10` to `<0.50` is review, and `>=0.50` is high. These are review triage colors, not universal release thresholds. |
| Standardize cards | Paired metric tiles always show real, synthetic, and gap in one tile. |
| Explain why differences occur | Advanced Evidence explains finite-sample estimation, sampling variation, rare-category grouping, and method/runtime trade-offs. |
| Test different distributions | The victim dataset tests demographics, offenses, and dates. The hospital dataset adds skewed length of stay, diagnosis mappings, payer, dispositions, and date-duration equations. |
| Make the page more information dense | Run summary, method, and findings were consolidated into a compact strip plus evidence sections. |
| State limits plainly | Advanced Evidence includes "What This Run Does Not Claim." |
| Improve the Windows UX | The old parameter sheet was replaced with four guided steps: choose data, select fields, configure, review results. |
| Let users select attributes | Every field can be included or omitted. The screen distinguishes modeled fields from replaced identifiers. |
| Show expected runtime | The estimate uses rows, modeled fields, method, CTGAN epochs, and optional repeated runs. It is calibrated from measured local runs. |
| Remove misleading methods | Baseline and empirical pattern matching are not shown in the primary app. |

## Fictitious Dataset Results

### Victim Data

- 20,000 source rows
- 5 modeled fields
- 4 identifier fields excluded and replaced
- SDV quality: approximately `0.988`
- Exact source identity matches: `0`
- Exact full modeled rows: `342` (`1.710%`)
- Rare tested source combinations reproduced: `336`
- Holdout DCR benchmark ratio: `0.959`
- Synthetic-to-real NNDR median: `0.716`
- Three-seed quality range: `0.9875` to `0.9881`
- Largest three-seed field-gap spread: `0.0068`
- Numeric/date/categorical gaps are shown individually in the dashboard

### Hospital Admissions

- 49,981 source rows
- 12 modeled fields
- 4 identifier fields excluded and replaced
- SDV quality: approximately `0.991`
- Column Pair Trends: approximately `0.991`
- Length-of-stay KS: approximately `0.0025`
- Exact source identity matches: `0`
- Exact full modeled rows: `1`
- Rare tested source combinations reproduced: `50`
- Holdout DCR benchmark ratio: `0.983`
- Synthetic-to-real NNDR median: `0.903`

The exact modeled row and rare-combination matches are not hidden. They are
reasons to review field selection, increase rare grouping, or add stronger
privacy mechanisms before a production release.

The distance ratios are empirical screens, not universal pass/fail thresholds.
The hospital ratio is close to parity but below `1.0`, so it remains a review
signal rather than being labeled a privacy pass.

## Useful Questions The Tool Enables

For victim/service data:

- Does the offense mix remain useful after direct identifiers are removed?
- Are age, race, sex, date, and offense relationships preserved?
- Which rare offense/demographic/date combinations reappear?
- Do conclusions remain stable across three random seeds?

For hospital data:

- Are admission type, payer, diagnosis, and disposition distributions preserved?
- Is length of stay preserved overall and by diagnosis?
- Are diagnosis code/description relationships still valid?
- Are admission/discharge date equations valid?
- Which selected attributes contribute most to exact or rare overlap?

For CAD data:

- Are call volume patterns preserved by day and hour?
- Are response-time distributions preserved by priority, call type, and
  neighborhood?
- Are units per event and event/unit grain preserved when unit data is available?
- Can researchers reproduce an aggregate finding when trained on synthetic data
  and tested on a protected real holdout?

## Current Privacy Boundary

This MVP does not provide a differential privacy epsilon and does not claim zero
re-identification risk. Community SDV quality metrics evaluate fidelity, not
privacy. The current privacy evidence is empirical:

- direct identifier source-value overlap
- exact source identity-combination overlap
- exact modeled-row overlap
- rare-combination exposure
- distance to the closest record, benchmarked against a real holdout
- nearest-neighbor distance ratio

Future production work should add agency-reviewed quasi-identifiers, calibrated
acceptance thresholds, targeted linkage attacks, and a privacy/utility
acceptance policy. Differential privacy should be evaluated separately rather
than implied by a high SDV quality score.
