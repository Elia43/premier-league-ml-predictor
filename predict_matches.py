# predict_matches.py
# Batch-predicts outcomes for a list of upcoming fixtures using the XGBoost model.

import pandas as pd
import numpy as np
import joblib

# ============================================================
# 1. Load model, encoders, and historical dataset
# ============================================================
MODEL_PATH      = "data/xgb_match_winner_model.pkl"
TEAM_ENC_PATH   = "data/team_encoder_xgb.pkl"
RESULT_ENC_PATH = "data/result_encoder_xgb.pkl"
DATASET_PATH    = "data/final_dataset_h2h.csv"

model          = joblib.load(MODEL_PATH)
team_encoder   = joblib.load(TEAM_ENC_PATH)
result_encoder = joblib.load(RESULT_ENC_PATH)

data = pd.read_csv(DATASET_PATH)
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
data = data.sort_values("Date").reset_index(drop=True)

# Feature order must match training exactly
# NOTE: "Unnamed: 0" is a CSV index artefact — NOT a training feature.
FEATURES = [
    "HomeTeam", "AwayTeam",
    "HTGS", "ATGS", "HTGC", "ATGC",
    "HTP", "ATP", "MW",
    "HTFormPts", "ATFormPts",
    "HTWinStreak3", "HTWinStreak5", "HTLossStreak3", "HTLossStreak5",
    "ATWinStreak3", "ATWinStreak5", "ATLossStreak3", "ATLossStreak5",
    "HTGD", "ATGD", "DiffPts", "DiffFormPts",
    "H2H_HomeWins", "H2H_AwayWins", "H2H_Draws",
    "H2H_HomeWinRate", "H2H_AwayWinRate",
    "Recent_H2H_HomeWinRate", "Recent_H2H_AwayWinRate",
]

AVAILABLE_TEAMS = sorted(set(data["HomeTeam"]).union(set(data["AwayTeam"])))

# ============================================================
# 2. Helper functions (same logic as predict_match.py)
# ============================================================

def encode_team(team):
    try:
        return team_encoder.transform([team])[0]
    except ValueError:
        return 0


def get_home_stats(team):
    home_games = data[data["HomeTeam"] == team]
    cols = ["HTGS", "HTGC", "HTP", "HTFormPts",
            "HTWinStreak3", "HTWinStreak5", "HTLossStreak3", "HTLossStreak5",
            "HTGD", "DiffPts", "DiffFormPts", "MW"]
    return home_games[cols].mean().fillna(0).to_dict() if len(home_games) > 0 else {c: 0 for c in cols}


def get_away_stats(team):
    away_games = data[data["AwayTeam"] == team]
    cols = ["ATGS", "ATGC", "ATP", "ATFormPts",
            "ATWinStreak3", "ATWinStreak5", "ATLossStreak3", "ATLossStreak5",
            "ATGD"]
    return away_games[cols].mean().fillna(0).to_dict() if len(away_games) > 0 else {c: 0 for c in cols}


def get_h2h_stats(home, away):
    matches = data[(data["HomeTeam"] == home) & (data["AwayTeam"] == away)]
    if len(matches) == 0:
        return {
            "H2H_HomeWins": 0, "H2H_AwayWins": 0, "H2H_Draws": 0,
            "H2H_HomeWinRate": 0.33, "H2H_AwayWinRate": 0.33,
            "Recent_H2H_HomeWinRate": 0.33, "Recent_H2H_AwayWinRate": 0.33
        }
    total     = len(matches)
    home_wins = (matches["FTR"] == "H").sum()
    away_wins = (matches["FTR"] == "A").sum()
    draws     = (matches["FTR"] == "D").sum()
    recent    = matches.tail(5)
    return {
        "H2H_HomeWins": home_wins, "H2H_AwayWins": away_wins, "H2H_Draws": draws,
        "H2H_HomeWinRate":        home_wins / total,
        "H2H_AwayWinRate":        away_wins / total,
        "Recent_H2H_HomeWinRate": (recent["FTR"] == "H").sum() / len(recent),
        "Recent_H2H_AwayWinRate": (recent["FTR"] == "A").sum() / len(recent),
    }


def build_feature_row(home, away, idx=0):
    return {
        "HomeTeam": encode_team(home),
        "AwayTeam": encode_team(away),
        **get_home_stats(home),
        **get_away_stats(away),
        **get_h2h_stats(home, away),
    }

# ============================================================
# 3. Batch prediction
# ============================================================

def predict_fixtures(fixtures: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Predict outcomes for a list of (home_team, away_team) tuples.

    Example:
        fixtures = [
            ("Arsenal",   "Chelsea"),
            ("Liverpool", "Man United"),
        ]
        results = predict_fixtures(fixtures)
        print(results)
    """
    rows, valid_fixtures = [], []

    for home, away in fixtures:
        if home not in AVAILABLE_TEAMS:
            print(f"⚠️  Skipping: '{home}' not in dataset.")
            continue
        if away not in AVAILABLE_TEAMS:
            print(f"⚠️  Skipping: '{away}' not in dataset.")
            continue
        rows.append(build_feature_row(home, away, idx=len(data) + len(rows)))
        valid_fixtures.append((home, away))

    if not rows:
        print("❌ No valid fixtures to predict.")
        return pd.DataFrame()

    X = pd.DataFrame(rows)[FEATURES]

    pred_encoded = model.predict(X)
    pred_labels  = result_encoder.inverse_transform(pred_encoded)
    probas       = model.predict_proba(X)
    classes      = list(result_encoder.classes_)

    results = []
    for i, (home, away) in enumerate(valid_fixtures):
        prob_map  = dict(zip(classes, probas[i]))
        home_prob = prob_map.get("H", prob_map.get("1", 0.0))
        draw_prob = prob_map.get("D", 0.0)
        away_prob = prob_map.get("A", prob_map.get("0", 0.0))
        label     = pred_labels[i]

        # Estimated scoreline
        hs = get_home_stats(home)
        as_ = get_away_stats(away)
        est_h = max(0, round(hs.get("HTGS", 1.0) * (home_prob + 0.5 * draw_prob)))
        est_a = max(0, round(as_.get("ATGS", 1.0) * (away_prob + 0.5 * draw_prob)))

        outcome_map = {"H": f"{home} Win", "D": "Draw", "A": f"{away} Win"}
        results.append({
            "Home Team":      home,
            "Away Team":      away,
            "Prediction":     outcome_map.get(label, label),
            "Home Win %":     f"{home_prob*100:.1f}%",
            "Draw %":         f"{draw_prob*100:.1f}%",
            "Away Win %":     f"{away_prob*100:.1f}%",
            "Est. Score":     f"{est_h} – {est_a}",
        })

    df_results = pd.DataFrame(results)
    return df_results


# ============================================================
# 4. Example usage – edit the fixtures list below
# ============================================================

if __name__ == "__main__":

    # ── Edit your upcoming fixtures here ──────────────────────
    upcoming_fixtures = [
        ("Arsenal",     "Chelsea"),
        ("Liverpool",   "Man United"),
        ("Man City",    "Tottenham"),
        ("Everton",     "Newcastle"),
    ]
    # ─────────────────────────────────────────────────────────

    print(f"\n📋  Available teams: {', '.join(AVAILABLE_TEAMS)}\n")

    results = predict_fixtures(upcoming_fixtures)

    if not results.empty:
        print("=" * 75)
        print("  ⚽  MATCH PREDICTIONS")
        print("=" * 75)
        print(results.to_string(index=False))
        print("=" * 75)

        # Optional: save to CSV
        save = input("\nSave predictions to CSV? (y/n): ").strip().lower()
        if save == "y":
            out_path = "data/predictions_output.csv"
            results.to_csv(out_path, index=False)
            print(f"✅ Saved to {out_path}")