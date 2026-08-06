# SyntheticCAD Presentation Brief

## One-Sentence Position

SyntheticCAD creates candidate synthetic data and an auditable evidence package
for agency review. It does not certify that data is risk-free, legally
unrestricted, or suitable for every research use.

## Five-Minute Story

### 1. The Problem

Police agencies and healthcare organizations hold useful operational data, but
sharing source records creates privacy, legal, security, and procurement
barriers. Researchers often cannot work inside the source environment.

### 2. The Product

The user selects a CSV, reviews automatically suggested field treatments,
chooses only the attributes needed for the research question, and runs synthesis
on the local Windows computer. SyntheticCAD produces candidate synthetic rows
and a review package with aggregate utility and privacy evidence.

### 3. What The Demo Proves

- Direct and record identifiers are excluded from model fitting and regenerated.
- Rare categories are grouped before fitting.
- SDV Gaussian Copula is the recommended first-run model.
- The dashboard separates statistical resemblance from privacy evidence.
- Exact matches, rare-combination exposure, and record-distance tails are shown
  with denominators and limitations.
- The shareable dashboard contains no real source rows or local file paths.

### 4. The Moat

SyntheticCAD is not trying to be another broad synthetic test-data platform.
The defensible wedge is a domain-aware release-review workflow:

- local operation inside the data owner's environment
- plain field treatment and explicit post-generation repair summaries
- CAD-specific event/unit relationship validation
- research-use utility questions, not only an overall similarity score
- an auditable evidence package with metric scope, formulas, versions, and
  non-claims
- a clear decision point: review, revise, or approve for a specific use

Broad platforms such as Syntho cover synthetic test-data management, masking,
connectors, and multiple deployment models. SyntheticCAD should differentiate
through public-safety release governance and domain evidence, not by claiming
that generic synthesis itself is novel.

### 5. The Honest Boundary

The current build is an evidence-generating MVP. It does not provide formal
differential privacy, zero re-identification risk, legal clearance, or an
independent holdout result. The present single-table pipeline also does not yet
prove the event-to-unit relationship required for full CAD scope.

### 6. The Next Proof

1. Fit only on a training split and evaluate fidelity and disclosure screens on
   a true holdout.
2. Run three seeds and report median and worst-tail results.
3. Build and validate the event/unit relational workflow on a real CAD schema.
4. Define one research utility task with an agency and researcher, then train on
   synthetic data and test on real holdout data.
5. Resolve SDV licensing, code signing, and installer size before commercial
   distribution.

## Demo Talk Track

Start on **Choose data** and say: "The source file stays on this computer."

On **Select fields**, say: "The operator chooses the minimum fields needed, and
identifier treatment is visible before generation."

On **Configure run**, say: "The first run is automatic. Gaussian Copula is the
recommended default; CTGAN is only an advanced comparison."

On **Basic Overview**, say: "This is decision support, not certification. The
top view shows the dataset and the specific evidence screens without hiding them
behind one score."

On **Advanced Evidence**, say: "Here are the quality components, metric scope,
formulas, denominators, lower tails, treatments, repairs, versions, and what the
run does not claim."

## Healthcare Expansion

Healthcare should be a separate domain pack built on the same evidence core,
not a generic rebrand. It would add healthcare-specific identifier detection,
date and length-of-stay constraints, code/description relationships, rare
diagnosis treatment, and use-case-specific utility tasks. CAD remains the first
domain proof because event/unit structure is central to the original product.

## Do Not Claim

- "The output is anonymous" or "risk-free."
- "A high SDV quality score proves privacy."
- "Zero exact matches means no re-identification risk."
- "The current distance screen is a protected holdout."
- "The single-table MVP has completed relational CAD support."
- "SyntheticCAD replaces agency counsel, privacy officers, or IRB review."
