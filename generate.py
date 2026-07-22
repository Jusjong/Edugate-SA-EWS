"""
SYNTHETIC South African EWS training dataset generator.

This dataset is entirely synthetic. No real student records, from any
institution, are used or resampled. Feature distributions and outcome
probabilities are calibrated against PUBLICLY PUBLISHED AGGREGATE
STATISTICS and effect sizes reported in the South African literature
(cited inline below and in methodology.md). Because no South African
institution publishes student-level microdata (POPIA + institutional
privacy), a statistically-calibrated synthetic generator is the only
"public data" route available — this script documents every calibration
choice so it can be re-run, audited, or replaced with real institutional
data under the same schema once available (see EWS Business Design
Document Section 6.1).

Calibration sources:
- NSFAS coverage ~61% of public university undergrads, ~80% of NSFAS
  recipients are first-years (NSFAS Vital Statistics 2022 / 2024 reporting).
- School quintile system: Q1-3 no-fee/under-resourced, Q4-5 fee-paying;
  Q1/Q2 associated with elevated first-year risk (Thusi et al. 2022;
  Kotze & Dreyer 2019).
- NSFAS funding disruption as an independent dropout risk factor;
  NSFAS coverage as protective factor at historically disadvantaged
  institutions (Branson 2024).
- DHET-reported dropout range ~30-40% within first three years for
  contact students (varies substantially by source/definition).
- APS alone misclassifies ~50% of at-risk first-year science students;
  NBT + NSC explain 35-45% of first-year outcome variance (Magagula
  et al. 2022; Van Zyl & Gravett 2020).
- Attendance and prior/prerequisite performance as strongest module-level
  predictors (Letseka et al. 2017; de Silva & Sanjula 2025).
- Ensemble methods on matric/quintile/NBT/programme achieving ~76%
  classification accuracy (Philippou, Ajoodha & Jadhav 2020) — used as
  the realism target for the baseline model trained at the end of this
  script.
- Qualification minimum-duration data (NQF level, minimum credits) for
  Bachelor of Commerce, Bachelor of Laws, and Higher Certificate in
  Management Development sourced from the SAQA National Learners'
  Records Database (NLRD) public register, University of the Free State,
  accessed via saqa.org.za qualification search (see qualifications_saqa.csv).
  Minimum duration is derived using the HEQSF convention of 120 credits
  per full-time-equivalent year (min_years = ceil(min_credits / 120)).
  Programmes without a specific SAQA lookup on file use the HEQSF
  standard duration for their qualification type (3-year bachelor's =
  360 credits; 4-year professional bachelor's = 480 credits), flagged
  as "HEQSF-standard" rather than "SAQA-verified" in the qualifications
  table — see qualifications_saqa.csv / README for how to extend this
  with further SAQA searches per faculty.

Run: python3 generate.py
Outputs (in ./output/): students.csv, student_years.csv,
module_enrollments.csv, weekly_engagement_sample.csv, data_dictionary.csv,
qualifications.csv
"""

import numpy as np
import pandas as pd
import uuid
import math
import os

rng = np.random.default_rng(42)
OUT = "output"
os.makedirs(OUT, exist_ok=True)

N_STUDENTS = 2000
COHORT_YEARS = [2021, 2022, 2023]

# ---------------------------------------------------------------------
# Qualification registry.
# "SAQA-verified" entries are taken directly from the SAQA NLRD public
# register (University of the Free State, accredited provider). All
# other entries use documented HEQSF-standard durations for their
# qualification type, pending a specific SAQA lookup.
# min_years = ceil(min_credits / 120), per HEQSF's 120-credits-per-FTE-year convention.
# ---------------------------------------------------------------------
QUALIFICATIONS = {
    "Bachelor of Commerce": dict(
        faculty="Commerce", department="Department of Business Management", saqa_id=8543, nqf_level=7, min_credits=536,
        qualification_type="National First Degree", source="SAQA-verified",
    ),
    "Higher Certificate in Management Development": dict(
        faculty="Commerce", department="School of Management", saqa_id=96676, nqf_level=5, min_credits=120,
        qualification_type="Higher Certificate", source="SAQA-verified",
    ),
    "Bachelor of Laws": dict(
        faculty="Law", department="Department of Private Law", saqa_id=110204, nqf_level=8, min_credits=480,
        qualification_type="National First Degree", source="SAQA-verified",
    ),
    "BSc Computer Science": dict(
        faculty="Science", department="Department of Computer Science and Informatics", saqa_id=None, nqf_level=7, min_credits=360,
        qualification_type="National First Degree", source="HEQSF-standard",
    ),
    "BSc Physics": dict(
        faculty="Science", department="Department of Physics", saqa_id=None, nqf_level=7, min_credits=360,
        qualification_type="National First Degree", source="HEQSF-standard",
    ),
    "BA Social Sciences": dict(
        faculty="Humanities", department="Department of Sociology", saqa_id=None, nqf_level=7, min_credits=360,
        qualification_type="National First Degree", source="HEQSF-standard",
    ),
    "BA Languages": dict(
        faculty="Humanities", department="Department of Language Studies", saqa_id=None, nqf_level=7, min_credits=360,
        qualification_type="National First Degree", source="HEQSF-standard",
    ),
    "BEng Civil": dict(
        faculty="Engineering", department="Department of Civil Engineering", saqa_id=None, nqf_level=8, min_credits=480,
        qualification_type="Professional Bachelor's Degree", source="HEQSF-standard",
    ),
    "BEng Electrical": dict(
        faculty="Engineering", department="Department of Electrical, Electronic and Computer Engineering", saqa_id=None, nqf_level=8, min_credits=480,
        qualification_type="Professional Bachelor's Degree", source="HEQSF-standard",
    ),
    "BSc Physiotherapy": dict(
        faculty="Health Sciences", department="School of Health and Rehabilitation Sciences", saqa_id=None, nqf_level=8, min_credits=480,
        qualification_type="Professional Bachelor's Degree", source="HEQSF-standard",
    ),
    "BSc Nursing": dict(
        faculty="Health Sciences", department="School of Nursing", saqa_id=None, nqf_level=8, min_credits=480,
        qualification_type="Professional Bachelor's Degree", source="HEQSF-standard",
    ),
    "BEd Foundation Phase": dict(
        faculty="Education", department="Department of Curriculum Studies", saqa_id=None, nqf_level=7, min_credits=480,
        qualification_type="Professional Bachelor's Degree", source="HEQSF-standard",
    ),
}
for _q in QUALIFICATIONS.values():
    _q["min_years"] = math.ceil(_q["min_credits"] / 120)

# Academic career: PeopleSoft/Campus-Solutions-style grouping above programme
# (Undergraduate / Postgraduate / Continuing Education). Constant here — see
# methodology.md — because this generator models first-time-entering
# undergraduates only (Section on "What this is"); it becomes a real,
# varying field once a postgraduate cohort is added.
ACADEMIC_CAREER = "Undergraduate"

PROGRAMME_NAMES = list(QUALIFICATIONS.keys())
PROGRAMME_WEIGHTS = [
    0.14,  # Bachelor of Commerce
    0.05,  # Higher Certificate in Management Development
    0.09,  # Bachelor of Laws
    0.09,  # BSc Computer Science
    0.07,  # BSc Physics
    0.09,  # BA Social Sciences
    0.08,  # BA Languages
    0.08,  # BEng Civil
    0.06,  # BEng Electrical
    0.08,  # BSc Physiotherapy
    0.08,  # BSc Nursing
    0.09,  # BEd Foundation Phase
]
assert abs(sum(PROGRAMME_WEIGHTS) - 1.0) < 1e-9

MODULES_PER_YEAR = {1: 5, 2: 5, 3: 4, 4: 3}

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ---------------------------------------------------------------------
# 1. STUDENTS (pre-entry, static)
# ---------------------------------------------------------------------
students = []
for i in range(N_STUDENTS):
    student_id = f"SYN{100000+i}"
    cohort_year = int(rng.choice(COHORT_YEARS))
    programme = str(rng.choice(PROGRAMME_NAMES, p=PROGRAMME_WEIGHTS))
    qual = QUALIFICATIONS[programme]
    faculty = qual["faculty"]

    # School quintile: university entrants skew toward higher quintiles
    # relative to the general school population (Q1-3 = ~75% of schools
    # nationally, but lower bachelor-pass rates reduce their share of
    # university entrants). Illustrative weighting, not a published
    # entrant-quintile breakdown (none is publicly available at this
    # granularity).
    quintile = int(rng.choice([1, 2, 3, 4, 5], p=[0.10, 0.13, 0.22, 0.25, 0.30]))

    first_gen = int(rng.random() < (0.62 if quintile <= 2 else 0.38 if quintile == 3 else 0.20))
    eal_flag = int(rng.random() < (0.75 if quintile <= 2 else 0.45 if quintile == 3 else 0.20))
    geo_origin = str(rng.choice(["rural", "urban"], p=[0.55, 0.45] if quintile <= 2 else [0.15, 0.85]))

    # NBT/NSC/APS: negatively associated with lower quintile, with noise
    quintile_penalty = (3 - quintile) * 4.5
    nbt_al = float(np.clip(rng.normal(62 - quintile_penalty, 12), 20, 100))
    nbt_ql = float(np.clip(rng.normal(55 - quintile_penalty * 1.3, 14), 10, 100))
    nbt_math = float(np.clip(rng.normal(50 - quintile_penalty * 1.4, 15), 5, 100))
    nsc_math = float(np.clip(rng.normal(58 - quintile_penalty, 16), 20, 100))
    nsc_physsci = float(np.clip(rng.normal(56 - quintile_penalty, 16), 20, 100))
    nsc_english = float(np.clip(rng.normal(60 - quintile_penalty * 0.6, 12), 25, 100))
    matric_aggregate = float(np.clip((nsc_math + nsc_physsci + nsc_english) / 3 + rng.normal(0, 5), 20, 100))
    aps_score = int(np.clip(round(matric_aggregate / 100 * 42 + rng.normal(0, 3)), 15, 42))

    # NSFAS eligibility strongly linked to quintile/household income proxy
    nsfas_eligible = int(rng.random() < (0.85 if quintile <= 2 else 0.60 if quintile == 3 else 0.25 if quintile == 4 else 0.08))

    students.append(dict(
        student_id=student_id, cohort_year=cohort_year, faculty=faculty,
        department=qual["department"], academic_career=ACADEMIC_CAREER, programme=programme,
        saqa_id=(qual["saqa_id"] if qual["saqa_id"] else ""), nqf_level=qual["nqf_level"],
        min_credits=qual["min_credits"], min_years_to_complete=qual["min_years"],
        qualification_source=qual["source"],
        school_quintile=quintile, first_gen_flag=first_gen, home_language_eal_flag=eal_flag,
        geographic_origin=geo_origin, nbt_academic_literacy=round(nbt_al, 1),
        nbt_quantitative_literacy=round(nbt_ql, 1), nbt_mathematics=round(nbt_math, 1),
        nsc_mathematics_pct=round(nsc_math, 1), nsc_physical_science_pct=round(nsc_physsci, 1),
        nsc_english_pct=round(nsc_english, 1), matric_aggregate_pct=round(matric_aggregate, 1),
        aps_score=aps_score, nsfas_eligible_initial=nsfas_eligible,
    ))

students_df = pd.DataFrame(students)

# Underlying (latent) academic-preparedness index, used to drive outcomes.
# NOT part of the exported schema — represents unobserved ability/readiness.
prep_index = (
    0.35 * (students_df["nbt_academic_literacy"] - 55) / 15
    + 0.30 * (students_df["nbt_quantitative_literacy"] - 50) / 15
    + 0.35 * (students_df["matric_aggregate_pct"] - 55) / 15
) + rng.normal(0, 0.6, N_STUDENTS)

# ---------------------------------------------------------------------
# 2. STUDENT-YEARS + 3. MODULE ENROLMENTS + 4. WEEKLY ENGAGEMENT SAMPLE
# ---------------------------------------------------------------------
student_years = []
module_enrollments = []
weekly_sample = []

WEEKLY_SAMPLE_COHORT = 2023  # only generate full weekly time-series for this cohort, to keep file size sane
WEEKLY_SAMPLE_MAX_STUDENTS = 120

weekly_sample_student_ids = set(
    students_df.loc[students_df.cohort_year == WEEKLY_SAMPLE_COHORT, "student_id"]
    .sample(n=WEEKLY_SAMPLE_MAX_STUDENTS, random_state=1)
)

for idx, srow in students_df.iterrows():
    sid = srow.student_id
    prep = prep_index[idx]
    quintile = srow.school_quintile
    faculty = srow.faculty
    max_years = QUALIFICATIONS[srow.programme]["min_years"]

    # persistent socio-economic/context state, refreshed per year (v1.7 design principle)
    nsfas_funded = bool(srow.nsfas_eligible_initial and rng.random() < 0.92)
    residence = str(rng.choice(["on_campus", "off_campus", "home"], p=[0.35, 0.35, 0.30]))

    active = True
    year_of_study = 1
    academic_year = srow.cohort_year
    total_credits_attempted = 0
    total_credits_earned = 0

    while active and year_of_study <= max_years + 2:  # allow up to 2 repeat years
        # --- annual context refresh (v1.7) ---
        disruption_p = 0.16 if nsfas_funded else 0.05
        funding_disruption = bool(nsfas_funded and rng.random() < disruption_p)
        if nsfas_funded and rng.random() < 0.05:
            nsfas_funded = False  # funding lapse this year (renewal failure)
        residence = residence if rng.random() > 0.10 else str(rng.choice(["on_campus", "off_campus", "home"]))

        # --- latent year risk score ---
        risk_z = (
            -0.9 * prep
            + 0.35 * (1 if quintile <= 2 else 0)
            + 0.45 * funding_disruption
            + 0.15 * srow.first_gen_flag
            + 0.10 * srow.home_language_eal_flag
            + 0.25 * max(0, year_of_study - max_years)  # already delayed -> compounding risk
            - 0.20 * (1 if nsfas_funded and not funding_disruption else 0)  # NSFAS protective when stable
            + rng.normal(0, 0.55)
        )
        fail_year_p = float(np.clip(sigmoid(risk_z - 1.1), 0.02, 0.85))
        dropout_p = float(np.clip(sigmoid(risk_z - 1.6) * (1.3 if year_of_study == 1 else 0.7), 0.01, 0.6))

        n_modules = MODULES_PER_YEAR.get(min(year_of_study, 4), 3)
        module_fail_flags = []

        for m in range(n_modules):
            module_code = f"{faculty[:3].upper()}{100*min(year_of_study,4) + m + 1}"
            session_type = str(rng.choice(["semester1", "semester2", "year"], p=[0.42, 0.42, 0.16]))

            attendance_base = float(np.clip(0.55 - 0.18 * risk_z + rng.normal(0, 0.12), 0.05, 1.0))
            early_att = float(np.clip(attendance_base + rng.normal(0, 0.08), 0, 1))
            early_logins = int(np.clip(rng.poisson(lam=max(0.5, 8 * attendance_base)), 0, 60))
            early_submission = float(np.clip(attendance_base * 0.9 + rng.normal(0, 0.1), 0, 1))

            consistency_penalty = float(np.clip(rng.normal(0.15 if risk_z > 0 else 0.05, 0.08), 0, 0.6))
            mid_att = float(np.clip(early_att - consistency_penalty + rng.normal(0, 0.06), 0, 1))
            mid_logins = int(np.clip(early_logins + rng.integers(-3, 4), 0, 80))
            mid_submission = float(np.clip(early_submission - consistency_penalty * 0.7, 0, 1))
            formative_avg = float(np.clip(55 - 14 * risk_z + rng.normal(0, 10), 0, 100))

            prereq_grade = None
            if year_of_study > 1 and rng.random() < 0.7:
                prereq_grade = float(np.clip(60 - 12 * risk_z + rng.normal(0, 12), 0, 100))

            module_risk_z = (
                0.9 * risk_z
                - 1.1 * (mid_att - 0.5)
                - 0.9 * (mid_submission - 0.5)
                - 0.02 * (formative_avg - 50)
                - (0.015 * ((prereq_grade or 60) - 50))
                + rng.normal(0, 0.5)
            )
            fail_p = float(np.clip(sigmoid(module_risk_z - 0.3), 0.02, 0.9))
            module_failed = int(rng.random() < fail_p)
            module_fail_flags.append(module_failed)
            final_mark = float(np.clip(
                (100 - 60 * fail_p) + rng.normal(0, 10) if not module_failed
                else rng.normal(35, 12), 0, 100
            ))
            outcome = "Fail" if module_failed else ("Withdrawn" if rng.random() < 0.02 else "Pass")

            enrollment_id = str(uuid.uuid4())[:8]
            module_enrollments.append(dict(
                enrollment_id=enrollment_id, student_id=sid, academic_year=academic_year,
                year_of_study=year_of_study, faculty=faculty, module_code=module_code,
                module_session_type=session_type,
                early_wave_attendance_pct=round(early_att * 100, 1),
                early_wave_lms_logins=early_logins,
                early_wave_submission_rate=round(early_submission, 2),
                mid_wave_attendance_pct=round(mid_att * 100, 1),
                mid_wave_lms_logins=mid_logins,
                mid_wave_submission_rate=round(mid_submission, 2),
                mid_wave_formative_avg=round(formative_avg, 1),
                engagement_consistency_score=round(1 - consistency_penalty, 2),
                prerequisite_grade=(round(prereq_grade, 1) if prereq_grade is not None else ""),
                final_mark=round(final_mark, 1),
                module_outcome=outcome,
                module_failure_label=module_failed,
            ))

            # weekly time series only for the illustrative sample subset
            if sid in weekly_sample_student_ids and year_of_study == 1:
                n_weeks = 13 if session_type != "year" else 26
                for wk in range(1, n_weeks + 1):
                    frac = wk / n_weeks
                    # engagement trends from early to mid/late wave, with weekly noise
                    att = early_att + (mid_att - early_att) * min(1, frac / 0.55) + rng.normal(0, 0.05)
                    logins = max(0, int(early_logins + (mid_logins - early_logins) * min(1, frac / 0.55) + rng.integers(-2, 3)))
                    weekly_sample.append(dict(
                        student_id=sid, academic_year=academic_year, module_code=module_code,
                        module_session_type=session_type, week_num=wk, session_fraction_elapsed=round(frac, 2),
                        lms_logins=logins, attendance_flag=int(rng.random() < np.clip(att, 0, 1)),
                        cumulative_submission_rate=round(float(np.clip(early_submission + (mid_submission - early_submission) * frac + rng.normal(0, 0.05), 0, 1)), 2),
                    ))

        module_fail_rate = np.mean(module_fail_flags) if module_fail_flags else 0
        year_failed = int((module_fail_rate > 0.5) or (rng.random() < fail_year_p * 0.4))
        gpa_year = float(np.clip(75 - 45 * module_fail_rate + rng.normal(0, 6), 30, 95))
        credits_attempted = n_modules * 12
        credits_earned = int(round(credits_attempted * (1 - module_fail_rate) * rng.uniform(0.9, 1.0)))
        total_credits_attempted += credits_attempted
        total_credits_earned += credits_earned
        credit_pace_ratio = (total_credits_earned / total_credits_attempted) if total_credits_attempted else 1.0

        dropped_out = int(rng.random() < (dropout_p + 0.25 * year_failed))

        standing = "exclusion" if (year_failed and year_of_study > max_years + 1) else \
                   "probation" if year_failed else "good"

        graduated = int(
            (not dropped_out) and year_of_study >= max_years
            and credit_pace_ratio > 0.80 and rng.random() < (0.55 if year_failed else 0.9)
        )

        year_outcome = "dropped_out" if dropped_out else "graduated" if graduated else "repeated" if year_failed else "progressed"

        student_years.append(dict(
            student_id=sid, academic_year=academic_year, year_of_study=year_of_study,
            gpa_year=round(gpa_year, 1), credits_attempted=credits_attempted, credits_earned=credits_earned,
            credit_pace_ratio=round(credit_pace_ratio, 2),
            nsfas_funded_this_year=int(nsfas_funded), funding_disruption_flag=int(funding_disruption),
            residence_status=residence, academic_standing=standing,
            year_outcome=year_outcome, year_failure_label=year_failed, dropout_label=dropped_out,
            # N+1 grace period, not any overrun at all: UFS General Academic Rules and
            # Regulations 2026, A21.9.1(a)(i) sets the honours-degree readmission review
            # point at "minimum duration... plus one (1) year" rather than the bare
            # minimum — a student finishing one year over minimum is normal, not delayed.
            delayed_graduation_label=int(graduated and year_of_study > max_years + 1),
        ))

        if dropped_out or graduated:
            active = False
        else:
            year_of_study += 1
            academic_year += 1

students_years_df = pd.DataFrame(student_years)
module_enrollments_df = pd.DataFrame(module_enrollments)
weekly_df = pd.DataFrame(weekly_sample)

students_df.to_csv(f"{OUT}/students.csv", index=False)
students_years_df.to_csv(f"{OUT}/student_years.csv", index=False)
module_enrollments_df.to_csv(f"{OUT}/module_enrollments.csv", index=False)
weekly_df.to_csv(f"{OUT}/weekly_engagement_sample.csv", index=False)

qual_rows = []
for name, q in QUALIFICATIONS.items():
    qual_rows.append(dict(
        programme=name, faculty=q["faculty"], department=q["department"], academic_career=ACADEMIC_CAREER,
        saqa_id=(q["saqa_id"] or ""), nqf_level=q["nqf_level"], min_credits=q["min_credits"],
        min_years_to_complete=q["min_years"], qualification_type=q["qualification_type"], source=q["source"],
    ))
pd.DataFrame(qual_rows).to_csv(f"{OUT}/qualifications.csv", index=False)

print(f"students: {len(students_df)} rows")
print(f"student_years: {len(students_years_df)} rows")
print(f"module_enrollments: {len(module_enrollments_df)} rows")
print(f"weekly_engagement_sample: {len(weekly_df)} rows")
print()
print("Outcome base rates:")
print("  module_failure_label:", round(module_enrollments_df.module_failure_label.mean(), 3))
print("  year_failure_label:  ", round(students_years_df.year_failure_label.mean(), 3))
print("  dropout_label:       ", round(students_years_df.dropout_label.mean(), 3))
print("  delayed_grad_label:  ", round(students_years_df.delayed_graduation_label.mean(), 3))

first_year_dropout = students_years_df[students_years_df.year_of_study == 1].dropout_label.mean()
print("  first-year dropout rate (calibration check, cf. DHET ~30-40%):", round(first_year_dropout, 3))
