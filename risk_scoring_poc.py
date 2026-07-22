"""
Risk scoring POC — calculation layer for the advisor/convenor UI prototype.

Computes three risk models using EXACT linear contribution decomposition
(logistic regression: each feature's contribution to the log-odds is
literally coef_i * standardized_x_i, so "top contributing factors" is
not an approximation here — it's the true local explanation for a
linear model). This is a faithful, lightweight stand-in for the SHAP
approach specified in EWS Business Design Document Section 4.5; swap
in a gradient-boosted model + real SHAP for production (see
validate_model.py for that variant).

Model A — Module-level risk (convenor/lecturer view):
  Features: mid-wave attendance, submission rate, formative average,
  engagement consistency, prerequisite grade.
  Label: module_failure_label.

Model B — Year-level failure risk (advisor view):
  Features: pre-entry academic (NBT, matric, APS) + persistent context
  (quintile, first-gen, EAL, NSFAS eligibility, funding disruption this
  year) + year_of_study, reflecting the v1.7 design principle that
  context/funding features persist across the study lifecycle while
  academic-prep features carry the pre-entry model. Trained across every
  student-year row (not just year 1), so it scores students at whatever
  year of study they're currently in — year_of_study is a legitimate
  feature here because year_failure_label is not defined in terms of it
  (the simulator's own risk formula includes a "+0.25 * years already
  delayed" compounding term, so this is real signal, not circularity).
  Label: year_failure_label.

Model C — Delayed-graduation risk (advisor view, new):
  Predicts the probability that a currently-enrolled student will
  eventually graduate later than their programme's minimum duration.
  IMPORTANT: generate.py defines delayed_graduation_label as
  int(graduated and year_of_study > max_years) — i.e. literally computed
  from year-of-study vs. minimum duration. Using "years enrolled so far"
  (or anything derived from it) as an input feature here would just let
  the model re-derive its own label's definition, not predict anything.
  So this model deliberately excludes year_of_study and is trained only
  on features that aren't algebraically tied to the label: pre-entry
  academic prep, persistent context, current credit_pace_ratio (a real,
  non-circular measure of how a student's cumulative progress is
  tracking), and the programme's min_years_to_complete. Trained on the
  ~460 students in the dataset who have already graduated (label known),
  scored against every currently-active student's latest year snapshot.
  "How long a student has been studying" is instead surfaced directly as
  a computed pace fact (year_of_study vs. min_years_to_complete) rather
  than folded into this score — see yearOfStudy/minYearsToComplete in
  the output and the honesty note in POC_HANDOFF.md.

Run: python3 risk_scoring_poc.py
Outputs: dashboard_data.json — feeds the HTML dashboard prototype
directly (embed as a <script> JSON blob, or fetch() it from a local
dev server).
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_DIR = "output"

def tier_of(p):
    if p >= 0.7:
        return "critical"
    if p >= 0.5:
        return "high"
    if p >= 0.3:
        return "medium"
    return "low"

# ---------------------------------------------------------------------
# Model A — module-level (convenor view)
# ---------------------------------------------------------------------
me = pd.read_csv(f"{DATA_DIR}/module_enrollments.csv")
me["prereq"] = pd.to_numeric(me["prerequisite_grade"], errors="coerce")
me["prereq"] = me["prereq"].fillna(me["prereq"].mean())

feat_a = ["mid_wave_attendance_pct", "mid_wave_submission_rate", "mid_wave_formative_avg",
          "engagement_consistency_score", "prereq"]
label_a = {
    "mid_wave_attendance_pct": "Attendance", "mid_wave_submission_rate": "Assignment submissions",
    "mid_wave_formative_avg": "Formative marks", "engagement_consistency_score": "Engagement consistency",
    "prereq": "Prerequisite grade",
}

Xa = me[feat_a].values
ya = me["module_failure_label"].values
scaler_a = StandardScaler().fit(Xa)
Xas = scaler_a.transform(Xa)
clf_a = LogisticRegression(max_iter=1000).fit(Xas, ya)
me["risk_proba"] = clf_a.predict_proba(Xas)[:, 1]

def contributions_a(row_pos):
    xs = Xas[row_pos]
    contribs = clf_a.coef_[0] * xs
    order = np.argsort(-np.abs(contribs))[:3]
    return [{"label": label_a[feat_a[i]], "direction": "increases" if contribs[i] > 0 else "decreases",
             "value": round(float(Xa[row_pos][i]), 1)} for i in order]

def modules_for(student_id, academic_year):
    """Every module a student is enrolled in during their current academic year,
    each scored by Model A — lets the advisor drawer show *which* module is
    driving an aggregate year-risk score, not just the aggregate itself."""
    rows = me[(me.student_id == student_id) & (me.academic_year == academic_year)].sort_values(
        "risk_proba", ascending=False
    )
    result = []
    for pos in rows.index:
        i = me.index.get_loc(pos)
        r = me.loc[pos]
        result.append({
            "module": r.module_code,
            "attendance": round(r.mid_wave_attendance_pct, 0),
            "submission": round(r.mid_wave_submission_rate * 100, 0),
            "formative": round(r.mid_wave_formative_avg, 0),
            "risk": round(float(r.risk_proba) * 100, 0),
            "tier": tier_of(r.risk_proba),
            "factors": contributions_a(i),
        })
    return result

top_module = me.module_code.value_counts().index[0]
sub = me[me.module_code == top_module].sort_values("risk_proba", ascending=False)
convenor_students = []
for pos in sub.head(14).index:
    i = me.index.get_loc(pos)
    row = me.loc[pos]
    convenor_students.append({
        "id": row.student_id, "module": row.module_code,
        "attendance": round(row.mid_wave_attendance_pct, 0),
        "submission": round(row.mid_wave_submission_rate * 100, 0),
        "formative": round(row.mid_wave_formative_avg, 0),
        "risk": round(float(row.risk_proba) * 100, 0),
        "tier": tier_of(row.risk_proba),
        "factors": contributions_a(i),
    })

# ---------------------------------------------------------------------
# Shared setup — students + full student_years history
# ---------------------------------------------------------------------
students = pd.read_csv(f"{DATA_DIR}/students.csv")
sy = pd.read_csv(f"{DATA_DIR}/student_years.csv").sort_values(["student_id", "academic_year"])

pre_entry_feats = ["school_quintile", "first_gen_flag", "home_language_eal_flag", "nbt_academic_literacy",
                    "nbt_quantitative_literacy", "nbt_mathematics", "matric_aggregate_pct", "aps_score",
                    "nsfas_eligible_initial"]

# Each student's most recent row — their "current" state for scoring the live caseload.
sy_latest = sy.groupby("student_id").tail(1).reset_index(drop=True)
active_latest = sy_latest[sy_latest.year_outcome.isin(["progressed", "repeated"])].copy()

# ---------------------------------------------------------------------
# Model B — year-level failure risk (advisor view)
# Trained across every student-year row, not just year 1, so it scores
# students at whatever year of study they're currently in.
# ---------------------------------------------------------------------
feat_b = pre_entry_feats + ["funding_disruption_flag", "year_of_study"]
label_b = {
    "school_quintile": "School quintile", "first_gen_flag": "First-generation student",
    "home_language_eal_flag": "English additional language", "nbt_academic_literacy": "NBT academic literacy",
    "nbt_quantitative_literacy": "NBT quantitative literacy", "nbt_mathematics": "NBT mathematics",
    "matric_aggregate_pct": "Matric aggregate", "aps_score": "APS score",
    "nsfas_eligible_initial": "NSFAS eligible", "funding_disruption_flag": "Funding disruption this year",
    "year_of_study": "Year of study",
}

train_b = sy.merge(students[["student_id"] + pre_entry_feats], on="student_id", how="inner")

Xb = train_b[feat_b].values
yb = train_b["year_failure_label"].values
scaler_b = StandardScaler().fit(Xb)
Xbs = scaler_b.transform(Xb)
clf_b = LogisticRegression(max_iter=1000).fit(Xbs, yb)

def contributions_b(xs):
    contribs = clf_b.coef_[0] * xs
    order = np.argsort(-np.abs(contribs))[:3]
    return [{"label": label_b[feat_b[i]], "direction": "increases" if contribs[i] > 0 else "decreases"} for i in order]

# ---------------------------------------------------------------------
# Model C — delayed-graduation risk (advisor view)
# Trained only on students who have already graduated (label is known),
# deliberately excluding year_of_study — see module docstring for why.
# ---------------------------------------------------------------------
feat_c = pre_entry_feats + ["credit_pace_ratio", "min_years_to_complete"]
label_c = {
    "school_quintile": "School quintile", "first_gen_flag": "First-generation student",
    "home_language_eal_flag": "English additional language", "nbt_academic_literacy": "NBT academic literacy",
    "nbt_quantitative_literacy": "NBT quantitative literacy", "nbt_mathematics": "NBT mathematics",
    "matric_aggregate_pct": "Matric aggregate", "aps_score": "APS score",
    "nsfas_eligible_initial": "NSFAS eligible", "credit_pace_ratio": "Cumulative credit pace",
    "min_years_to_complete": "Programme minimum duration",
}

grad_rows = sy[sy.year_outcome == "graduated"]
train_c = grad_rows.merge(
    students[["student_id", "min_years_to_complete"] + pre_entry_feats], on="student_id", how="inner"
)

Xc = train_c[feat_c].values
yc = train_c["delayed_graduation_label"].values
scaler_c = StandardScaler().fit(Xc)
Xcs = scaler_c.transform(Xc)
clf_c = LogisticRegression(max_iter=1000).fit(Xcs, yc)

def contributions_c(xs):
    contribs = clf_c.coef_[0] * xs
    order = np.argsort(-np.abs(contribs))[:3]
    return [{"label": label_c[feat_c[i]], "direction": "increases" if contribs[i] > 0 else "decreases"} for i in order]

# ---------------------------------------------------------------------
# Score the live, currently-active caseload on both Model B and Model C
# ---------------------------------------------------------------------
active = active_latest.merge(
    students[["student_id", "programme", "min_years_to_complete"] + pre_entry_feats], on="student_id", how="inner"
).reset_index(drop=True)

Xb_active = scaler_b.transform(active[feat_b].values)
active["year_failure_proba"] = clf_b.predict_proba(Xb_active)[:, 1]

Xc_active = scaler_c.transform(active[feat_c].values)
active["delayed_grad_proba"] = clf_c.predict_proba(Xc_active)[:, 1]

high = active.sort_values("year_failure_proba", ascending=False).head(9)
low = active.sort_values("year_failure_proba", ascending=True).head(3)
caseload = pd.concat([high, low])

advisor_students = []
for pos in caseload.index:
    row = caseload.loc[pos]
    advisor_students.append({
        "id": row.student_id, "programme": row.programme, "quintile": int(row.school_quintile),
        "nsfas": bool(row.nsfas_eligible_initial), "fundingDisruption": bool(row.funding_disruption_flag),
        "yearOfStudy": int(row.year_of_study), "minYearsToComplete": int(row.min_years_to_complete),
        "risk": round(float(row.year_failure_proba) * 100, 0), "tier": tier_of(row.year_failure_proba),
        "factors": contributions_b(Xb_active[pos]),
        "delayedGradRisk": round(float(row.delayed_grad_proba) * 100, 0),
        "delayedGradTier": tier_of(row.delayed_grad_proba),
        "delayedGradFactors": contributions_c(Xc_active[pos]),
        "modules": modules_for(row.student_id, row.academic_year),
    })

out = {
    "convenor": {"module": top_module, "students": convenor_students},
    "advisor": {"students": advisor_students},
}
with open("dashboard_data.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"Convenor view: {len(convenor_students)} students in module {top_module}")
print(f"Advisor view: {len(advisor_students)} students in caseload sample "
      f"(spanning years of study: {sorted(caseload.year_of_study.unique().tolist())})")
print(f"Model C (delayed graduation) trained on {len(train_c)} graduated students "
      f"({yc.mean():.1%} delayed)")
print("Written: dashboard_data.json")
