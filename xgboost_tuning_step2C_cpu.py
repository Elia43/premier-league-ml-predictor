import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

# 1️⃣ Load the dataset
data = pd.read_csv("data/final_dataset_h2h.csv")
data = data.sort_values("Date").reset_index(drop=True)

# 2️⃣ Encode categorical columns
team_encoder = LabelEncoder()
for col in ["HomeTeam", "AwayTeam"]:
    data[col] = team_encoder.fit_transform(data[col])

# Encode the match result
result_encoder = LabelEncoder()
data["FTR"] = result_encoder.fit_transform(data["FTR"])


# 3️⃣ Define features and target
features = data.columns.drop(["FTR"], errors="ignore")
X = data[features]
y = data["FTR"]

leakage_cols = ["FTHG", "FTAG"]
X = X.drop(columns=[c for c in leakage_cols if c in X.columns])


# Drop non-numeric columns
drop_cols = [
    "Date", "HM1", "HM2", "HM3", "HM4", "HM5",
    "AM1", "AM2", "AM3", "AM4", "AM5",
    "HTFormPtsStr", "ATFormPtsStr"
]
X = X.drop(columns=[c for c in drop_cols if c in X.columns])



# 4️⃣ Split chronologically (train before test)
split_index = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# 5️⃣ Define the XGBoost model
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',        # ← CPU
    predictor='cpu_predictor', # ← CPU
    use_label_encoder=False
)

# 6️⃣ Grid Search for tuning
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1
)

print("🔍 Starting XGBoost tuning...")
grid_search.fit(X_train, y_train)

# 7️⃣ Get best model
best_model = grid_search.best_estimator_
print(f"\n✅ Best Parameters: {grid_search.best_params_}")

# 8️⃣ Evaluate
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 📊 Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix – XGBoost Match Prediction")
plt.show()

# 📈 ROC–AUC
from sklearn.metrics import roc_auc_score, RocCurveDisplay

y_proba = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
print(f"📈 ROC–AUC Score: {roc_auc:.4f}")

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve – XGBoost Match Prediction")
plt.show()

corrs = data.corr(numeric_only=True)["FTR"].sort_values(ascending=False)
print(corrs)


# 9️⃣ Save model and encoders
joblib.dump(best_model, "data/xgb_match_winner_model.pkl")
joblib.dump(team_encoder, "data/team_encoder_xgb.pkl")
joblib.dump(result_encoder, "data/result_encoder_xgb.pkl")

print("\n✅ Model and encoders saved in /data folder.")