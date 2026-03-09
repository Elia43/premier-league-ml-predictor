import pandas as pd

# === Step 1: Load your existing dataset ===
data = pd.read_csv("data/final_dataset.csv")

# Make sure it’s sorted chronologically
if "Date" in data.columns:
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")

# Ensure the key columns exist
required_cols = {"HomeTeam", "AwayTeam", "FTR"}
if not required_cols.issubset(data.columns):
    raise ValueError(f"Dataset missing required columns: {required_cols - set(data.columns)}")

# === Step 2: Create an empty list to store results ===
records = []

# === Step 3: Dictionary to store past match results ===
history = {}

for _, row in data.iterrows():
    home = row["HomeTeam"]
    away = row["AwayTeam"]

    pair = tuple(sorted([home, away]))  # (A,B) same as (B,A)
    past_matches = history.get(pair, [])

    # Compute H2H stats from past matches
    if len(past_matches) == 0:
        h2h_home_wins = h2h_away_wins = h2h_draws = 0
    else:
        df_past = pd.DataFrame(past_matches)
        h2h_home_wins = (df_past["winner"] == home).sum()
        h2h_away_wins = (df_past["winner"] == away).sum()
        h2h_draws = (df_past["winner"] == "D").sum()

    total_h2h = max(1, len(past_matches))
    h2h_home_rate = h2h_home_wins / total_h2h
    h2h_away_rate = h2h_away_wins / total_h2h

    # Recent (last 5 matches) rates
    recent = df_past.tail(5) if len(past_matches) > 0 else pd.DataFrame(columns=["winner"])
    if not recent.empty:
        rec_home_rate = (recent["winner"] == home).sum() / len(recent)
        rec_away_rate = (recent["winner"] == away).sum() / len(recent)
    else:
        rec_home_rate = rec_away_rate = 0

    # Save computed features
    record = {
        "H2H_HomeWins": h2h_home_wins,
        "H2H_AwayWins": h2h_away_wins,
        "H2H_Draws": h2h_draws,
        "H2H_HomeWinRate": h2h_home_rate,
        "H2H_AwayWinRate": h2h_away_rate,
        "Recent_H2H_HomeWinRate": rec_home_rate,
        "Recent_H2H_AwayWinRate": rec_away_rate,
    }
    records.append(record)

    # === Update history with this match ===
    if row["FTR"] == "H":
        winner = home
    elif row["FTR"] == "A":
        winner = away
    else:
        winner = "D"

    history.setdefault(pair, []).append({"winner": winner})

# === Step 4: Merge new features with original dataset ===
h2h_df = pd.DataFrame(records)
data_h2h = pd.concat([data.reset_index(drop=True), h2h_df], axis=1)

# === Step 5: Save updated dataset ===
output_path = "data/final_dataset_h2h.csv"
data_h2h.to_csv(output_path, index=False)
print(f"✅ H2H features added and saved to: {output_path}")
print("✅ New columns added:", list(h2h_df.columns))
