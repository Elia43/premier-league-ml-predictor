# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ======================
# 1️⃣ Load dataset
# ======================
data = pd.read_csv("data/final_dataset.csv")

# ======================
# 2️⃣ Encode categorical team names
# ======================
# This lets the model understand that "Liverpool" ≠ "Luton"
team_encoder = LabelEncoder()
data['HomeTeam'] = team_encoder.fit_transform(data['HomeTeam'])
data['AwayTeam'] = team_encoder.transform(data['AwayTeam'])

# ======================
# 3️⃣ Encode target result (FTR)
# ======================p
# FTR = Full Time Result → H = Home Win, A = Away Win, D = Draw
result_encoder = LabelEncoder()
data['FTR'] = result_encoder.fit_transform(data['FTR'])

# ======================
# 4️⃣ Select features (added HomeTeam + AwayTeam)
# ======================
features = [
    'HomeTeam', 'AwayTeam',    # NEW: helps model identify teams
    'HTGS', 'ATGS',            # Goals scored so far
    'HTGC', 'ATGC',            # Goals conceded so far
    'HTP', 'ATP',              # Points up to that match
    'HTGD', 'ATGD',            # Goal difference
    'DiffPts', 'DiffFormPts'   # Differences between teams
]

X = data[features]
y = data['FTR']

# ======================
# 5️⃣ Split the data
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# 6️⃣ Train the model
# ======================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ======================
# 7️⃣ Evaluate performance
# ======================
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=result_encoder.classes_))

# ======================
# 8️⃣ Save model and encoders
# ======================
joblib.dump(model, "data/match_winner_model.pkl")
joblib.dump(result_encoder, "data/result_encoder.pkl")
joblib.dump(team_encoder, "data/team_encoder.pkl")

print("\nModel and encoders saved successfully!")

