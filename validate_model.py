"""
Baseline model validation — confirms the synthetic dataset is trainable
and produces realistic performance in the range reported by the cited
SA literature (~75-85% accuracy / AUC for module and pre-entry
classifiers). This is a sanity check on the generator, not a production
model.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

OUT = "output"

# ---------------------------------------------------------------------
# Model A: Module-level failure prediction (mid-wave features)
# ---------------------------------------------------------------------
me = pd.read_csv(f"{OUT}/module_enrollments.csv")
features_mid = [
    "early_wave_attendance_pct", "early_wave_lms_logins", "early_wave_submission_rate",
    "mid_wave_attendance_pct", "mid_wave_lms_logins", "mid_wave_submission_rate",
    "mid_wave_formative_avg", "engagement_consistency_score",
]
me["prerequisite_grade_filled"] = pd.to_numeric(me["prerequisite_grade"], errors="coerce")
me["has_prerequisite"] = me["prerequisite_grade_filled"].notna().astype(int)
me["prerequisite_grade_filled"] = me["prerequisite_grade_filled"].fillna(me["prerequisite_grade_filled"].mean())
features_mid += ["prerequisite_grade_filled", "has_prerequisite"]

X = me[features_mid]
y = me["module_failure_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model_a = GradientBoostingClassifier(random_state=42)
model_a.fit(X_train, y_train)
proba = model_a.predict_proba(X_test)[:, 1]
pred = model_a.predict(X_test)
print("=== Model A: Module failure (mid-session wave) — Gradient Boosting ===")
print("AUC:     ", round(roc_auc_score(y_test, proba), 3))
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("Base rate (fail):", round(y.mean(), 3))
importances = sorted(zip(features_mid, model_a.feature_importances_), key=lambda t: -t[1])
print("Top features:", [(f, round(v, 3)) for f, v in importances[:5]])
print()

# ---------------------------------------------------------------------
# Model A-early: same, but early-wave-only features (screening signal)
# ---------------------------------------------------------------------
features_early = ["early_wave_attendance_pct", "early_wave_lms_logins", "early_wave_submission_rate"]
Xe = me[features_early]
Xe_train, Xe_test, ye_train, ye_test = train_test_split(Xe, y, test_size=0.25, random_state=42, stratify=y)
model_a_early = GradientBoostingClassifier(random_state=42)
model_a_early.fit(Xe_train, ye_train)
proba_e = model_a_early.predict_proba(Xe_test)[:, 1]
print("=== Model A-early: Module failure (early wave only) ===")
print("AUC:     ", round(roc_auc_score(ye_test, proba_e), 3))
print("(Expected: lower than mid-wave model — matches de Silva & Sanjula 2025 finding that")
print(" predictive power improves markedly from week 5 onward.)")
print()

# ---------------------------------------------------------------------
# Model B: Pre-entry model (student-level, year-1 features only)
# ---------------------------------------------------------------------
students = pd.read_csv(f"{OUT}/students.csv")
sy = pd.read_csv(f"{OUT}/student_years.csv")
y1 = sy[sy.year_of_study == 1][["student_id", "year_failure_label", "dropout_label"]]
pre = students.merge(y1, on="student_id", how="inner")

features_pre = [
    "school_quintile", "first_gen_flag", "home_language_eal_flag",
    "nbt_academic_literacy", "nbt_quantitative_literacy", "nbt_mathematics",
    "nsc_mathematics_pct", "nsc_physical_science_pct", "nsc_english_pct",
    "matric_aggregate_pct", "aps_score", "nsfas_eligible_initial",
]
Xp = pre[features_pre]
yp = pre["year_failure_label"]
Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.25, random_state=42, stratify=yp)

model_b = RandomForestClassifier(n_estimators=300, random_state=42)
model_b.fit(Xp_train, yp_train)
proba_b = model_b.predict_proba(Xp_test)[:, 1]
pred_b = model_b.predict(Xp_test)
print("=== Model B: Pre-entry year-1 failure prediction — Random Forest (bagging) ===")
print("AUC:     ", round(roc_auc_score(yp_test, proba_b), 3))
print("Accuracy:", round(accuracy_score(yp_test, pred_b), 3))
print("(Reference point: Philippou, Ajoodha & Jadhav 2020 report 75.97% accuracy with")
print(" bagging on matric/quintile/NBT/programme features on synthetic Bayesian-network data.)")
print()

# APS-only baseline, to reproduce the Magagula et al. (2022) "APS alone is unreliable" finding
model_aps = LogisticRegression(max_iter=1000)
model_aps.fit(Xp_train[["aps_score"]], yp_train)
pred_aps = model_aps.predict(Xp_test[["aps_score"]])
print("=== APS-only baseline (for comparison) ===")
print("Accuracy:", round(accuracy_score(yp_test, pred_aps), 3))
print("Misclassification rate:", round(1 - accuracy_score(yp_test, pred_aps), 3),
      "(cf. Magagula et al. 2022: ~50% misclassification using APS alone)")
