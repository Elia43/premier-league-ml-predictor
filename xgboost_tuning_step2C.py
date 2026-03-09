# xgboost_tuning_step2C.py
# XGBoost training with GridSearchCV hyperparameter tuning.
# Automatically uses GPU (CUDA) if available, falls back to CPU gracefully.
# Target: true 3-class prediction — Home Win (H), Draw (D), Away Win (A).

import subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no blocking plt.show()
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection  import GridSearchCV
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import (accuracy_score, classification_report,
                                       confusion_matrix, ConfusionMatrixDisplay,
                                       roc_auc_score)
from xgboost import XGBClassifier

# ============================================================
# 0. GPU auto-detection
# ============================================================
def _has_gpu() -> bool:
    """Return True if nvidia-smi reports at least one CUDA-capable GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

DEVICE = "cuda" if _has_gpu() else "cpu"
print(f"⚙️   Running on: {DEVICE.upper()}")

# ============================================================
# 1. Load dataset
# ============================================================
data = pd.read_csv("data/final_dataset_h2h.csv")
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
data = data.sort_values("Date").reset_index(drop=True)

# ============================================================
# 2. Encode teams with a SINGLE shared encoder (fit once on ALL teams)
# ============================================================
all_teams = pd.concat([data["HomeTeam"], data["AwayTeam"]]).unique()
team_encoder = LabelEncoder()
team_encoder.fit(all_teams)                         # ← fit once, transform twice

data["HomeTeam"] = team_encoder.transform(data["HomeTeam"])
data["AwayTeam"] = team_encoder.transform(data["AwayTeam"])

# ============================================================
# 3. Rebuild the 3-class target correctly from raw scorelines
#    Original FTR was binary (H / NH) — this restores H / D / A
# ============================================================
def build_ftr(row):
    if row["FTHG"] > row["FTAG"]:
        return "H"
    elif row["FTHG"] < row["FTAG"]:
        return "A"
    else:
        return "D"

data["FTR"] = data.apply(build_ftr, axis=1)

result_encoder = LabelEncoder()
data["FTR"] = result_encoder.fit_transform(data["FTR"])
print(f"🎯  Classes: {list(result_encoder.classes_)}")

# ============================================================
# 4. Build feature matrix — drop leakage and non-numeric columns
# ============================================================
DROP_LEAKAGE     = ["FTHG", "FTAG"]
DROP_NON_NUMERIC = [
    "Date",
    "HM1", "HM2", "HM3", "HM4", "HM5",
    "AM1", "AM2", "AM3", "AM4", "AM5",
    "HTFormPtsStr", "ATFormPtsStr",
    "Unnamed: 0",                          # CSV index artefact — not a feature
]
DROP_ALL = DROP_LEAKAGE + DROP_NON_NUMERIC + ["FTR"]

feature_cols = [c for c in data.columns if c not in DROP_ALL]
X = data[feature_cols]
y = data["FTR"]

print(f"📐  Feature matrix: {X.shape[0]} rows × {X.shape[1]} columns")

# ============================================================
# 5. Chronological 80/20 train / test split (no shuffle — time series)
# ============================================================
split = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ============================================================
# 6. XGBoost model — GPU if available, CPU fallback
# ============================================================
model = XGBClassifier(
    objective         = "multi:softmax",   # true 3-class objective
    num_class         = 3,
    eval_metric       = "mlogloss",        # multi-class log-loss
    tree_method       = "hist",            # works on both CPU and GPU (XGBoost >= 1.7)
    device            = DEVICE,            # "cuda" or "cpu" — set automatically above
    use_label_encoder = False,
    random_state      = 42,
)

# ============================================================
# 7. GridSearchCV hyperparameter tuning (5-fold)
# ============================================================
param_grid = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample":     [0.8, 1.0],
}

grid_search = GridSearchCV(
    estimator  = model,
    param_grid = param_grid,
    cv         = 5,
    scoring    = "accuracy",
    verbose    = 2,
    n_jobs     = 1 if DEVICE == "cuda" else -1,  # GPU already parallelises internally
)

print("\n🔍  Starting XGBoost hyperparameter tuning …")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\n✅  Best Parameters : {grid_search.best_params_}")

# ============================================================
# 8. Evaluation
# ============================================================
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯  Test Accuracy   : {accuracy:.4f}")
print("\nClassification Report:\n",
      classification_report(y_test, y_pred,
                             target_names=result_encoder.classes_))

# ROC-AUC (One-vs-Rest, handles 3 classes)
roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
print(f"📈  Macro ROC-AUC   : {roc_auc:.4f}")

# ============================================================
# 9. Save charts to disk (no plt.show() — runs cleanly on servers/GPU machines)
# ============================================================

# 9a. Confusion matrix
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=result_encoder.classes_)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix — XGBoost Match Prediction")
fig.tight_layout()
fig.savefig("data/confusion_matrix.png", dpi=150)
plt.close(fig)
print("💾  Saved: data/confusion_matrix.png")

# 9b. Feature importance
importance_df = (
    pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": best_model.feature_importances_,
    })
    .sort_values("Importance", ascending=False)
    .head(20)
)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(importance_df["Feature"], importance_df["Importance"], color="#2196F3")
ax.invert_yaxis()
ax.set_xlabel("Importance")
ax.set_title("Top 20 Feature Importances — XGBoost")
fig.tight_layout()
fig.savefig("data/feature_importance.png", dpi=150)
plt.close(fig)
print("💾  Saved: data/feature_importance.png")

# 9c. Feature–result correlation heatmap (top 10 features)
top10 = importance_df["Feature"].head(10).tolist()
corr_data = data[top10 + ["FTR"]].corr(numeric_only=True)[["FTR"]].drop("FTR")

fig, ax = plt.subplots(figsize=(4, 6))
im = ax.imshow(corr_data.values, aspect="auto", cmap="RdYlGn",
               vmin=-0.3, vmax=0.3)
ax.set_yticks(range(len(corr_data)))
ax.set_yticklabels(corr_data.index, fontsize=9)
ax.set_xticks([0])
ax.set_xticklabels(["Correlation with Result"])
plt.colorbar(im, ax=ax)
ax.set_title("Feature–Result Correlation (Top 10)")
fig.tight_layout()
fig.savefig("data/feature_correlation.png", dpi=150)
plt.close(fig)
print("💾  Saved: data/feature_correlation.png")

# ============================================================
# 10. Save model, encoders, and feature column list
# ============================================================
joblib.dump(best_model,    "data/xgb_match_winner_model.pkl")
joblib.dump(team_encoder,  "data/team_encoder_xgb.pkl")
joblib.dump(result_encoder,"data/result_encoder_xgb.pkl")
joblib.dump(feature_cols,  "data/feature_columns_xgb.pkl")   # aligned inference

print("\n✅  Model, encoders, and feature list saved to /data")
print(f"    Device used : {DEVICE.upper()}")
print(f"    Accuracy    : {accuracy:.4f}")
print(f"    ROC-AUC     : {roc_auc:.4f}")