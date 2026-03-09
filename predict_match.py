# predict_match.py
# Predicts the outcome of a single football match using the best-trained XGBoost model.

import pandas as pd
import numpy as np
import joblib

# ============================================================
# 1. Load the best model (XGBoost) and encoders
# ============================================================
MODEL_PATH       = "data/xgb_match_winner_model.pkl"
TEAM_ENC_PATH    = "data/team_encoder_xgb.pkl"
RESULT_ENC_PATH  = "data/result_encoder_xgb.pkl"
DATASET_PATH     = "data/final_dataset_h2h.csv"

model          = joblib.load(MODEL_PATH)
team_encoder   = joblib.load(TEAM_ENC_PATH)
result_encoder = joblib.load(RESULT_ENC_PATH)

data = pd.read_csv(DATASET_PATH)
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
data = data.sort_values("Date").reset_index(drop=True)

# Exact feature order the XGBoost model was trained on
# NOTE: "Unnamed: 0" is a CSV index artefact — the model was NOT trained with it.
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

# ============================================================
# 2. Helper functions
# ============================================================

def get_available_teams():
    return sorted(set(data["HomeTeam"]).union(set(data["AwayTeam"])))


def validate_team(team):
    if team not in get_available_teams():
        print(f"❌ Team '{team}' not found.")
        print(f"   Available teams: {', '.join(get_available_teams())}")
        return False
    return True


def encode_team(team):
    """Encode a team name. Falls back to 0 if unseen (should not happen after validation)."""
    try:
        return team_encoder.transform([team])[0]
    except ValueError:
        print(f"⚠️  Warning: '{team}' not in encoder. Using fallback value 0.")
        return 0


def get_home_stats(team):
    """Average stats from all home games played by this team."""
    home_games = data[data["HomeTeam"] == team]
    cols = ["HTGS", "HTGC", "HTP", "HTFormPts",
            "HTWinStreak3", "HTWinStreak5", "HTLossStreak3", "HTLossStreak5",
            "HTGD", "DiffPts", "DiffFormPts", "MW"]
    return home_games[cols].mean().fillna(0).to_dict() if len(home_games) > 0 else {c: 0 for c in cols}


def get_away_stats(team):
    """Average stats from all away games played by this team."""
    away_games = data[data["AwayTeam"] == team]
    cols = ["ATGS", "ATGC", "ATP", "ATFormPts",
            "ATWinStreak3", "ATWinStreak5", "ATLossStreak3", "ATLossStreak5",
            "ATGD"]
    return away_games[cols].mean().fillna(0).to_dict() if len(away_games) > 0 else {c: 0 for c in cols}


def get_goals_per_game(team):
    """
    Per-game scoring averages using raw FTHG / FTAG columns.
    HTGS / ATGS are CUMULATIVE season totals — using them directly
    in score estimates produces absurd results (e.g. 19-15).
    This function uses the actual goals per match instead.
    """
    home_games = data[data["HomeTeam"] == team]
    away_games = data[data["AwayTeam"] == team]

    home_scored  = home_games["FTHG"].mean() if len(home_games) > 0 else 1.2
    away_scored  = away_games["FTAG"].mean() if len(away_games) > 0 else 1.0
    home_conceded = home_games["FTAG"].mean() if len(home_games) > 0 else 1.2
    away_conceded = away_games["FTHG"].mean() if len(away_games) > 0 else 1.4

    return {
        "home_scored":   round(home_scored,  3),   # avg goals scored at home
        "away_scored":   round(away_scored,  3),   # avg goals scored away
        "home_conceded": round(home_conceded, 3),  # avg goals conceded at home
        "away_conceded": round(away_conceded, 3),  # avg goals conceded away
    }


def get_h2h_stats(home, away):
    """Head-to-head stats strictly from home vs away perspective."""
    matches = data[(data["HomeTeam"] == home) & (data["AwayTeam"] == away)]

    if len(matches) == 0:
        # No direct history – use neutral priors
        return {
            "H2H_HomeWins": 0, "H2H_AwayWins": 0, "H2H_Draws": 0,
            "H2H_HomeWinRate": 0.33, "H2H_AwayWinRate": 0.33,
            "Recent_H2H_HomeWinRate": 0.33, "Recent_H2H_AwayWinRate": 0.33
        }

    total         = len(matches)
    home_wins     = (matches["FTR"] == "H").sum()
    away_wins     = (matches["FTR"] == "A").sum()
    draws         = (matches["FTR"] == "D").sum()

    recent        = matches.tail(5)
    rec_home_rate = (recent["FTR"] == "H").sum() / len(recent)
    rec_away_rate = (recent["FTR"] == "A").sum() / len(recent)

    return {
        "H2H_HomeWins":            home_wins,
        "H2H_AwayWins":            away_wins,
        "H2H_Draws":               draws,
        "H2H_HomeWinRate":         home_wins / total,
        "H2H_AwayWinRate":         away_wins / total,
        "Recent_H2H_HomeWinRate":  rec_home_rate,
        "Recent_H2H_AwayWinRate":  rec_away_rate,
    }


# ============================================================
# 3. Main prediction function
# ============================================================

def predict_match(home_team: str, away_team: str) -> dict | None:
    """
    Predict the outcome of a match.

    Returns a dict with keys:
        predicted_result, home_win_prob, draw_prob, away_win_prob, predicted_score
    """
    if not validate_team(home_team) or not validate_team(away_team):
        return None
    if home_team == away_team:
        print("❌ Home and away team cannot be the same.")
        return None

    print(f"\n{'='*50}")
    print(f"  ⚽  {home_team}  (H)  vs  {away_team}  (A)")
    print(f"{'='*50}")

    # Build feature row — order must match FEATURES list exactly
    row = {
        "HomeTeam": encode_team(home_team),
        "AwayTeam": encode_team(away_team),
        **get_home_stats(home_team),
        **get_away_stats(away_team),
        **get_h2h_stats(home_team, away_team),
    }

    df_input = pd.DataFrame([row])[FEATURES]

    # ── Predict ──────────────────────────────────────────────
    pred_encoded = model.predict(df_input)[0]
    pred_label   = result_encoder.inverse_transform([pred_encoded])[0]
    proba        = model.predict_proba(df_input)[0]
    classes      = list(result_encoder.classes_)

    prob_map  = dict(zip(classes, proba))
    home_prob = prob_map.get("H", prob_map.get("1", 0.0))
    draw_prob = prob_map.get("D", 0.0)
    away_prob = prob_map.get("A", prob_map.get("0", 0.0))

    # ── Score estimate using per-game goals (NOT cumulative HTGS/ATGS) ──
    # Blend each team's attack average with opponent's defensive weakness
    # to get a realistic expected goals figure, then round to nearest int.
    home_gpg = get_goals_per_game(home_team)
    away_gpg = get_goals_per_game(away_team)

    # expected goals = avg(team attack, opponent defence allowed) × win-weight
    raw_home = (home_gpg["home_scored"] + away_gpg["away_conceded"]) / 2
    raw_away = (away_gpg["away_scored"] + home_gpg["home_conceded"]) / 2

    # Scale by outcome probability — winning team scores more, losing team less
    est_h = max(0, round(raw_home * (home_prob + 0.5 * draw_prob) * 2))
    est_a = max(0, round(raw_away * (away_prob + 0.5 * draw_prob) * 2))

    # Sanity-cap at 6 goals per side (heuristic, not a hard rule)
    est_h = min(est_h, 6)
    est_a = min(est_a, 6)

    # ── Print results ─────────────────────────────────────────
    print(f"\n📊  WIN PROBABILITIES")
    print(f"    🏠 {home_team:<20} {home_prob*100:5.1f}%")
    print(f"    🤝 Draw                    {draw_prob*100:5.1f}%")
    print(f"    ✈️  {away_team:<20} {away_prob*100:5.1f}%")

    outcome_map = {"H": f"🏠 {home_team} Win", "D": "🤝 Draw", "A": f"✈️  {away_team} Win"}
    print(f"\n🏆  Predicted Result : {outcome_map.get(pred_label, pred_label)}")
    print(f"⚽  Estimated Score  : {home_team} {est_h} – {est_a} {away_team}\n")

    return {
        "predicted_result": pred_label,
        "home_win_prob":    round(home_prob, 4),
        "draw_prob":        round(draw_prob, 4),
        "away_win_prob":    round(away_prob, 4),
        "predicted_score":  f"{est_h}-{est_a}",
    }


# ============================================================
# 4. Interactive CLI
# ============================================================

if __name__ == "__main__":
    teams = get_available_teams()
    print(f"\n📋  Available teams ({len(teams)} total):")
    for i, t in enumerate(teams, 1):
        print(f"   {i:>2}. {t}")

    print()
    home = input("Enter Home Team: ").strip()
    away = input("Enter Away Team: ").strip()
    predict_match(home, away)