"""
Trains a churn classifier on the real Telco Customer Churn dataset (IBM
Sample Data Sets, via Kaggle), using the SAME cleaning function that the
API uses at inference time (clean_customers from data_prep.py) — this
avoids the classic train/serve skew bug where cleaning logic drifts apart
between notebook and production code.
"""
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay

from data_prep import clean_customers

# All files (data, model, charts) are saved right next to this script, so
# there's no folder structure to worry about — this works no matter what
# the repo folder is named or where it's cloned/extracted to.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "telco_customer_churn_raw.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.joblib")
OUT_DIR = SCRIPT_DIR

df = pd.read_csv(DATA_PATH, encoding="utf-8")
df = clean_customers(df, is_training=True)

feature_cols = [c for c in df.columns if c != "Churn"]
X = df[feature_cols]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print(f"Dataset: {len(df)} customers | churn rate: {y.mean():.1%}")
print(f"Test AUC: {auc:.3f}\n")
print(classification_report(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("Top 10 feature importances:")
print(importances.head(10))

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
ax.set_title(f"Churn Model ROC Curve (AUC = {auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=150)

fig2, ax2 = plt.subplots(figsize=(7, 5))
importances.head(10).sort_values().plot.barh(ax=ax2, color="#3b6ea5")
ax2.set_title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)

# Save the model AND the exact feature column order, so inference can't drift
joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)
print(f"\nSaved model -> {MODEL_PATH}")
print(f"Saved charts -> {OUT_DIR}/roc_curve.png, {OUT_DIR}/feature_importance.png")
