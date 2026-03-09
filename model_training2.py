# model_training_step2A.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === 1. Load and prepare data ===
data = pd.read_csv("data/final_dataset.csv")

# Parse dates safely (handles 00-style years)
data['Date'] = pd.to_datetime(data['Date'], format='mixed', dayfirst=True, errors='coerce')
data = data.sort_values('Date').reset_index(drop=True)

# Create time-based features
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Season'] = data['Date'].dt.year

# === 2. Encode categorical columns ===
team_encoder = LabelEncoder()
data['HomeTeamEnc'] = team_encoder.fit_transform(data['HomeTeam'])
data['AwayTeamEnc'] = team_encoder.fit_transform(data['AwayTeam'])

# === 3. Encode target (FTR) ===
result_encoder = LabelEncoder()
data['FTR_Encoded'] = result_encoder.fit_transform(data['FTR'])

# === 4. Build Head-to-Head (H2H) features (FIXED – no data leakage) ===
h2h_home_rate = []
h2h_away_rate = []
match_history = {}

for i, row in data.iterrows():
    home = row['HomeTeam']
    away = row['AwayTeam']
    key_home = (home, away)

    # Retrieve past meetings only
    past_meetings = match_history.get(key_home, [])
    if len(past_meetings) > 0:
        home_wins = past_meetings.count('H')
        away_wins = past_meetings.count('A')
        total = len(past_meetings)
        h2h_home_rate.append(home_wins / total)
        h2h_away_rate.append(away_wins / total)
    else:
        h2h_home_rate.append(0.5)
        h2h_away_rate.append(0.5)

    # Update match history AFTER calculating
    match_result_label = result_encoder.inverse_transform([row['FTR_Encoded']])[0]
    match_history.setdefault(key_home, []).append(match_result_label)

data['H2H_HomeWinRate'] = h2h_home_rate
data['H2H_AwayWinRate'] = h2h_away_rate

# === 5. Select features for training ===
features = [
    'HomeTeamEnc', 'AwayTeamEnc', 'Month', 'DayOfWeek', 'Season',
    'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP',
    'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts',
    'H2H_HomeWinRate', 'H2H_AwayWinRate'
]

X = data[features]
y = data['FTR_Encoded']

# === 6. Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# === 7. Train model ===
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# === 8. Evaluate ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=result_encoder.classes_))

# === 9. Save model and encoders ===
joblib.dump(model, "data/match_winner_model_step2A.pkl")
joblib.dump(result_encoder, "data/result_encoder_step2A.pkl")
joblib.dump(team_encoder, "data/team_encoder_step2A.pkl")

print("\n✅ Model and encoders with Head-to-Head features saved successfully!")
