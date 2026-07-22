# Synthetic EWS Training Dataset — Methodology & Disclaimer

## What this is

A fully synthetic dataset of 2,000 simulated students, matched to the schema in Section 6.1 of the *EWS Business Design Document*, for use in prototyping and testing the risk-scoring models described in Sections 4.3–4.5. It is intended to let model development, feature pipelines, and dashboards be built and tested **before** real institutional data is available, and to serve as a reference implementation of the feature engineering described in the design.

## What this is not

**No real student records were used, sampled, or resampled.** No South African institution publishes student-level microdata — this is a POPIA and institutional-privacy constraint, not an oversight, and it is exactly the constraint Section 7 of the design document is built around. There is therefore no real "public dataset" of South African student records to anonymise. Instead, this generator produces synthetic students from scratch, with feature distributions and outcome probabilities **calibrated against publicly published aggregate statistics and effect sizes** from South African research and government/NSFAS reporting. Every calibration choice is listed below so it can be audited, challenged, or replaced.

**Do not present outputs of models trained on this data as findings about real students.** Its purpose is engineering validation (does the pipeline work, is the schema right, do the models train) — not a substitute for Phase 1 (data audit) and Phase 3 (real-cohort training) of the design's phased rollout.

## Calibration sources

| Parameter | Calibration | Source |
|---|---|---|
| First-year dropout rate | ~30% (simulated) | DHET-reported range ~30-40% for contact students within 3 years; other sources cite higher figures depending on definition |
| NSFAS coverage | ~61% of undergraduates; ~80% of recipients are first-years | NSFAS Vital Statistics reporting (2022/2024) |
| Quintile effect on preparedness | Q1/Q2 students shifted ~9-13 points lower on NBT/NSC scales | Thusi et al. (2022); Kotze & Dreyer (2019) — quintile as proxy for systemic deprivation |
| NSFAS as protective factor / disruption as risk | Funded+stable = protective; disruption = independent risk spike | Branson (2024) |
| APS-alone unreliability | Full-feature model outperforms APS-only baseline | Magagula et al. (2022) — APS alone misclassifies ~50% of at-risk first-years |
| Ensemble accuracy target (pre-entry) | ~75% target range | Philippou, Ajoodha & Jadhav (2020) — bagging, 75.97% |
| Attendance/prerequisite grade as top module predictors | Attendance and prerequisite grade weighted most heavily in module failure simulation | Letseka et al. (2017); de Silva & Sanjula (2025) |
| Engagement consistency vs. volume | Consistency score included as a distinct feature from login counts | de Silva & Sanjula (2025) |
| Early-wave vs. mid-wave predictive power | Early-wave model built weaker by design | de Silva & Sanjula (2025) — reliable detection from week 5, plateau by week 8 |
| Programme minimum duration | Bachelor of Commerce (536 credits, 5 yrs), Bachelor of Laws (480 credits, 4 yrs), Higher Certificate in Management Development (120 credits, 1 yr) | **SAQA National Learners' Records Database, public qualification register, University of the Free State (accredited provider)** — see `qualifications_saqa.csv` and `qualifications.csv`. Other programmes use documented HEQSF-standard durations (360 credits/3 yrs for a standard bachelor's, 480 credits/4 yrs for a professional bachelor's), flagged `HEQSF-standard` rather than `SAQA-verified` pending further register lookups |

Where no public breakdown exists at the required granularity (e.g., exact quintile distribution of *university entrants* specifically, as opposed to the general school population), the generator uses an illustrative, clearly-marked assumption rather than an invented precise figure — see comments in `generate.py`.

## Files produced

- `students.csv` — 2,000 rows, one per student, pre-entry features plus qualification/programme metadata
- `student_years.csv` — ~6,000 rows, one per student per year of study, including funding/context refresh and the three student-level outcome labels
- `module_enrollments.csv` — ~26,000 rows, one per student per module per year, with early-wave and mid-wave features and the module-level outcome label
- `weekly_engagement_sample.csv` — ~9,000 rows, full weekly time series for a 120-student illustrative subset (2023 cohort, year 1 only), showing the shape of the continuous scoring data described in Section 4.3
- `qualifications.csv` — 12 programmes with NQF level, minimum credits, minimum duration, and provenance (SAQA-verified vs. HEQSF-standard)
- `qualifications_saqa.csv` — the raw extracted records from the SAQA NLRD search (source of truth for the 3 SAQA-verified programmes)
- `data_dictionary.csv` — field-level documentation for all five tables

## Qualification data — SAQA source

Minimum programme duration is not an arbitrary modelling choice — it is the definitional basis for the `delayed_graduation_label` ("graduates later than minimum degree duration," Section 3 of the design). Three programmes in this dataset use real, publicly registered data from the **SAQA National Learners' Records Database (NLRD)**, the statutory public register of all qualifications registered on South Africa's National Qualifications Framework (saqa.org.za), filtered to University of the Free State as accredited provider:

| Programme | SAQA ID | NQF Level | Min. Credits | Min. Years (credits/120) |
|---|---|---|---|---|
| Bachelor of Commerce | 8543 | 7 | 536 | 5 |
| Bachelor of Laws | 110204 | 8 | 480 | 4 |
| Higher Certificate in Management Development | 96676 | 5 | 120 | 1 |

(The SAQA search also returned two postgraduate diplomas — Estate Planning and Transfusion Medicine, both 120 credits/1 year — retained in `qualifications_saqa.csv` for completeness but not used in the current simulation, which models first-time-entering undergraduates only.)

**Department and academic career (hierarchy fields).** `students.csv` and `qualifications.csv` now carry `department` (academic department/school within the faculty) and `academic_career` (Undergraduate/Postgraduate/Continuing Education grouping above programme), so the advisor view can navigate Faculty → Department → Academic Career → Programme → Student. `academic_career` is a constant `"Undergraduate"` — honestly, not artificially — because this generator models first-time-entering undergraduates only; it becomes a real, varying field once a postgraduate cohort is added. Department names for the two Health Sciences programmes (School of Nursing, School of Health and Rehabilitation Sciences) were confirmed against ufs.ac.za; the remaining ten use typical, illustrative department naming for their faculty and were not individually verified against a register the way `qualification_source` is — this is a materially lower confidence level and should not be conflated with the SAQA-verified/HEQSF-standard provenance already tracked for programme duration.

**Delayed-graduation threshold is minimum duration + 1 year, not the bare minimum.** `delayed_graduation_label` in `generate.py` flags a student only once `year_of_study > min_years_to_complete + 1` at graduation — finishing one year over minimum is treated as normal, not delayed. This follows the **University of the Free State General Academic Rules and Regulations (2026)**, rule A21.9.1(a)(i), which sets the honours-degree readmission review point at "the minimum duration of the relevant qualification... plus one (1) year" rather than the bare minimum (the same document's A12.11(j) sets a separate, harder *maximum allowable* period at minimum + 2 years for undergraduate bachelor's degrees before academic exclusion — not currently modelled as a distinct threshold, since only one delay signal was in scope here).

The remaining 9 programmes (Science, Engineering, Health Sciences, Education, and the rest of Humanities) do not yet have a specific SAQA register lookup on file, so they use documented **HEQSF-standard** durations by qualification type — 360 credits/3 years for a standard bachelor's degree, 480 credits/4 years for a professional bachelor's degree (Engineering, Health Sciences, Education) — clearly flagged `qualification_source = "HEQSF-standard"` in `students.csv` and `qualifications.csv` so the two provenance levels are never conflated. **To extend this to full SAQA coverage**, run additional NLRD searches (search.php on saqa.org.za) filtered to the accredited provider of interest, for the Science, Engineering, Humanities, and Education fields, and add the results to `qualifications_saqa.csv` in the same format — `generate.py`'s `QUALIFICATIONS` dict is the single place to update.

## Baseline model validation

A baseline model was trained on this dataset to confirm it produces trainable, realistic signal (not a production model — see `validate_model.py`):

| Model | Result | Literature reference point |
|---|---|---|
| Module failure, mid-session wave (Gradient Boosting) | AUC 0.83, accuracy 79% | Cf. 75-85% range reported across cited SA/international studies |
| Module failure, early wave only | AUC 0.79 (lower than mid-wave) | Matches de Silva & Sanjula's finding that predictive power improves from week 5 |
| Pre-entry year-1 failure (Random Forest / bagging) | Accuracy 72% | Cf. Philippou et al.'s 75.97% with a similar feature set |
| APS-only baseline | Accuracy 66% (34% misclassified) | Directionally consistent with Magagula et al.'s ~50% misclassification finding; the full-feature model clearly outperforms it |

Top features for module failure (feature importance): mid-wave attendance, mid-wave formative average, mid-wave submission rate — consistent with the design's Section 4.5 rationale for weighting behavioural and formative signals over background alone.

## How to use this with real data later

The table and field names deliberately mirror the design document's Section 6.1 entities (`students`, `student_terms`→`student_years`, `course_enrollments`→`module_enrollments`, `lms_engagement`→`weekly_engagement_sample`). When real institutional data becomes available (Phase 0/1 of the rollout), the same model training code should run against real extracts with minimal changes — that parity is the point of building the synthetic set to this schema.

## Regenerating or adjusting

Run `python3 generate.py` (requires `pandas`, `numpy`) to regenerate with the same seed (reproducible) or change `rng = np.random.default_rng(42)` for a different draw. Key parameters (`N_STUDENTS`, quintile weights, risk-model coefficients) are declared near the top of the script and commented with their rationale.
