# EWS Risk Dashboard POC — handoff to Claude Code

## Goal

A proof-of-concept visual dashboard showing calculated student risk to two
audiences, per the *EWS Business Design Document* (see project folder):

- **Convenor/lecturer view** — module-level risk (Section 4.5): which students
  in a specific module are at risk of failing it, with top contributing factors.
- **Advisor view** — year-level risk (Section 4.3): a caseload of students
  ranked by risk tier, with top contributing factors, for cross-module/
  cross-year student support.

Scope for this POC: **calculation + display only**. Data ingestion, SIS/LMS
integration, and the read-only mirror architecture (Section 4.6) are assumed
already solved — out of scope here.

## What's already done (in this folder)

- `generate.py` — builds the full synthetic SA student dataset (2,000
  students, calibrated against public SA statistics; see `methodology.md`).
  Includes real SAQA-registered qualification data for 3 UFS programmes.
- `output/` — the generated CSVs (students, student_years, module_enrollments,
  weekly_engagement_sample, qualifications).
- `validate_model.py` — trains Gradient Boosting / Random Forest models on
  the dataset and confirms it produces literature-consistent results (module
  AUC ~0.82, pre-entry accuracy ~72%).
- `risk_scoring_poc.py` — the calculation layer for this POC specifically.
  Trains two **logistic regression** models (module-level and year-level)
  and computes an exact linear contribution decomposition per student
  (coefficient × standardized feature value = that feature's literal
  contribution to the risk score — a faithful, lightweight stand-in for the
  SHAP explanations specified in Section 4.5, chosen here because it's exact
  for a linear model rather than approximate).
- `dashboard_data.json` — the output of `risk_scoring_poc.py`: 14 students
  for a convenor view of module `COM102`, and a 12-student caseload sample
  (9 highest-risk + 3 lowest-risk, for contrast) for the advisor view. Each
  student record includes `risk` (0-100), `tier` (critical/high/medium/low),
  and `factors` (top 3 contributing signals with direction).

## Design principles the UI should follow (derived from the design doc)

1. **Explainability is adjacent to the score, not buried** — the top 3
   contributing factors must be visible next to every risk number, not
   behind a click.
2. **Tier is the primary visual hierarchy** — colour + text label together
   (never colour alone), consistent between the convenor and advisor views.
3. **Action over analytics** — the dominant call-to-action on a flagged
   student should be "log intervention" / "mark contacted", not just "view
   detail". The design's whole premise (Sections 4.4, 5, 13) is that
   identification without action doesn't move outcomes.
4. **Two distinct audiences, one visual language** — convenor view scopes to
   one module; advisor view scopes to a cross-module caseload. Same tier
   colours and card layout, different data shape.

## Suggested next step in Claude Code

Build a small local web app (single HTML file is fine for a POC, or a
minimal React/Vite app if you want it to grow) that:

- Reads `dashboard_data.json`.
- Has a toggle/tab between "Convenor — COM102" and "Advisor — caseload".
- Renders each student as a card: id, risk score, tier badge, top 3 factors
  (with direction arrows), and a "Log intervention" button (can be a stub —
  no backend needed for the POC).
- Sorts by risk descending by default.

The JSON is already shaped for direct consumption — no further data
wrangling should be needed to get a first visual working.

## Regenerating the data

If qualification, feature, or calibration changes are needed, edit
`generate.py` (documented, see `methodology.md`), then re-run:

```
python3 generate.py
python3 risk_scoring_poc.py
```
