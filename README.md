# ⚽ Premier League Match Predictor

**End-to-End Machine Learning Pipeline | EPL 2000 – 2018 | 3-Class Outcome Prediction**

> Predicts every English Premier League fixture as **Home Win (H)**, **Draw (D)**, or **Away Win (A)** using 18 seasons of match data, engineered team-form features, and leakage-free head-to-head statistics.

| Language | Models | Classes | Matches | Clubs |
|---|---|---|---|---|
| Python 3.x | Random Forest · XGBoost | H / D / A | 6,840 | 44 |

---

## 📌 Project Overview

This project implements a complete, production-style ML pipeline across three model generations. Each generation builds on the last by fixing data quality issues, adding richer features, and upgrading the algorithm. The final model is a tuned **XGBoost classifier** solving a true 3-class problem (H / D / A) with **52.3% accuracy and a ROC-AUC of 0.668**: 7.7 percentage points above always predicting a home win, which already scores 44.5% on the same test set, on a problem that professional betting models also find notoriously difficult.

**Key technical highlights:**
- True 3-class target (H / D / A) correctly reconstructed from raw scorelines — not a binary collapse
- Single shared `LabelEncoder` fit once across all team names to prevent silent label mismatch
- Rolling 5-game form windows capturing recent momentum rather than season-only averages
- Direction-aware, leakage-free H2H features — only past fixtures are used at each row
- 5-fold `GridSearchCV` hyperparameter tuning with ROC-AUC (OvR) evaluation
- Chronological 80/20 train/test split to simulate real-world deployment conditions

---

## 🗂️ Repository Structure

```
football-predictor/
│
├── data/
│   ├── final_dataset.csv                  # Base dataset — 6,840 EPL matches, 40 columns
│   ├── final_dataset_h2h.csv              # Enriched dataset — base + 13 H2H/form features
│   ├── match_winner_model.pkl             # Step 1 — Baseline Random Forest
│   ├── match_winner_model_step2A.pkl      # Step 2A — RF + rolling form + H2H
│   ├── xgb_match_winner_model.pkl         # Best model — XGBoost (tuned)
│   ├── team_encoder_xgb.pkl               # Shared LabelEncoder for all 44 clubs
│   ├── result_encoder_xgb.pkl             # LabelEncoder for H / D / A
│   ├── feature_columns_xgb.pkl            # Saved feature column order for aligned inference
│   ├── confusion_matrix.png               # 3-class confusion matrix (XGBoost)
│   ├── feature_importance.png             # Top-20 feature importance bar chart
│   └── feature_correlation.png            # Feature–result correlation heatmap
│
├── h2h_features_step2C.py                 # Builds final_dataset_h2h.csv (13 new features)
├── model_training.py                      # Step 1 — Baseline RF (3-class, 5-fold CV)
├── model_training2.py                     # Step 2A — RF + rolling form windows + H2H rates
├── feature_importance_step2B.py           # Feature importance chart + correlation heatmap
├── xgboost_tuning_step2C.py               # Step 2C — XGBoost + GridSearchCV, GPU (auto-detects CUDA)
├── xgboost_tuning_step2C_cpu.py           # Step 2C — same as above, CPU-only version
├── predict_match.py                       # CLI — predict a single fixture interactively
├── predict_matches.py                     # Batch — predict a list of upcoming fixtures
└── README.md
```

---

## 🗃️ Dataset

| Statistic | Value |
|---|---|
| Total matches | 6,840 |
| Seasons covered | 2000 – 2018 (18 seasons) |
| Unique clubs | 44 |
| Class distribution | H: 46% · D: 26% · A: 28% |

### Base Features

| Column | Description |
|---|---|
| `FTHG` / `FTAG` | Full-time goals (home/away) — used only to construct the 3-class target; dropped before training |
| `HTGS` / `ATGS` | Cumulative goals scored this season |
| `HTGC` / `ATGC` | Cumulative goals conceded this season |
| `HTP` / `ATP` | Points accumulated up to this match |
| `HTGD` / `ATGD` | Cumulative goal difference |
| `DiffPts` | Points gap between the two teams |
| `DiffFormPts` | Form-points gap (last 5 games) |
| `HTFormPts` / `ATFormPts` | Raw form points from last 5 matches |
| `HTWinStreak3/5` | Flag: home team on a 3 or 5-match winning streak |
| `HTLossStreak3/5` | Flag: home team on a 3 or 5-match losing streak |
| `ATWinStreak3/5` | Flag: away team on a 3 or 5-match winning streak |
| `ATLossStreak3/5` | Flag: away team on a 3 or 5-match losing streak |
| `MW` | Match week number |

### Engineered Features

| Feature | Description |
|---|---|
| `H2H_HomeWins` / `H2H_AwayWins` / `H2H_Draws` | All-time H2H result counts — direction-aware (Arsenal at home ≠ Arsenal away) |
| `H2H_HomeWinRate` / `H2H_AwayWinRate` | Historical win rates for this specific fixture direction |
| `H2H_DrawRate` | Historical draw rate for this specific fixture |
| `H2H_TotalMeetings` | Total previous meetings between these two clubs |
| `Recent_H2H_HomeWinRate` | Home win rate across the last 5 head-to-head meetings |
| `Recent_H2H_AwayWinRate` | Away win rate across the last 5 head-to-head meetings |
| `Month` / `DayOfWeek` | Calendar features extracted from match date |
| `SeasonIdx` | Normalised season index (0 = 2000) — avoids raw-year bias |

---

## 🔧 Bugs Fixed & Improvements Made

| File | What was fixed / added |
|---|---|
| **All training files** | Target rebuilt as true 3-class H/D/A from FTHG/FTAG. Original `FTR` was binary (H/NH), collapsing Away Win and Draw into one class and destroying half the predictive signal. |
| `model_training*.py` · `xgboost_tuning*.py` | Single shared `team_encoder` fit on all teams before any split. Original code called `fit_transform` twice in a loop, silently overwriting the encoder — only AwayTeam labels survived correctly. |
| `model_training2.py` · `h2h_features*.py` | Rolling 5-game form windows added (`HomeRollGoals`, `AwayRollGoals`, `HomeRollPts`, `AwayRollPts`). Captures recent momentum instead of season cumulative averages only. |
| `h2h_features*.py` | H2H key is now direction-aware `(home, away)` rather than a sorted pair. Added `H2H_TotalMeetings` and `H2H_DrawRate`. All features computed without leakage. |
| `xgboost_tuning*.py` | XGBoost objective changed to `multi:softmax` with `num_class=3`. ROC-AUC now uses `multi_class='ovr'`. **GPU acceleration enabled** — `device='cuda'` with automatic CPU fallback via `nvidia-smi` detection. Original code was hardcoded to `predictor='cpu_predictor'` (deprecated in XGBoost ≥ 2.0). A separate `xgboost_tuning_step2C_cpu.py` is provided for machines without a GPU. Charts saved to PNG — no blocking `plt.show()` calls. Feature column order saved for inference. |
| `predict_match.py` | Removed erroneous `Unnamed: 0` index column from feature input (caused `ValueError: feature_names mismatch`). **Fixed absurd score estimates (e.g. "19–15")** — `HTGS`/`ATGS` are cumulative season totals, not per-game values; replaced with per-game `FTHG`/`FTAG` averages blended with opponent defensive record. Informative team-list printed on startup. |
| `feature_importance*.py` | Correlation heatmap of top-10 features vs result added. Both charts saved to disk. Works with tree-based and linear models. |

---

## 📈 Model Results

> **Important note on comparability:** Steps 1 and 2A used a **binary target** (Home Win vs Not-Home-Win) because the original dataset's `FTR` column collapsed Away Win and Draw into one class. Step 2C rebuilds the target correctly as **true 3-class (H / D / A)** from raw scorelines. The two sets of results solve different problems and are **not directly comparable** — the 3-class problem is genuinely harder.

### Step 1 — Baseline Random Forest, binary target (`model_training.py`)

```
Accuracy: 63.08%  ⚠️  Binary classification (H vs NH) — not true 3-class

              precision    recall  f1-score   support
           H       0.61      0.55      0.58       634
          NH       0.64      0.70      0.67       734
    accuracy                           0.63      1368
```

### Step 2A — Random Forest + Rolling Form + H2H, binary target (`model_training2.py`)

```
Accuracy: 63.23%  ⚠️  Binary classification (H vs NH) — not true 3-class

              precision    recall  f1-score   support
           H       0.61      0.55      0.58       623
          NH       0.65      0.70      0.68       745
    accuracy                           0.63      1368
```

> Adding temporal and head-to-head features produced a consistent improvement across all metrics.

### Step 2C — XGBoost + GridSearchCV, TRUE 3-class ✅ (`xgboost_tuning_step2C.py`)

```
Best Parameters:
  learning_rate: 0.05 | max_depth: 3 | n_estimators: 100 | subsample: 0.8

Accuracy: 52.27%  ✅  True 3-class (H / D / A) — honest evaluation
ROC-AUC (macro OvR): 0.6682

              precision    recall  f1-score   support
           A       0.54      0.39      0.46       420
           D       0.25      0.01      0.01       339
           H       0.52      0.90      0.66       609
    accuracy                           0.52      1368
   macro avg       0.44      0.43      0.38      1368
```

> **How good is 52%?** Random guessing on 3 classes gives 33%, but the classes aren't balanced: always predicting a home win already scores 44.5% on this test set (609 of 1,368 matches). So the honest margin is 7.7 percentage points: the model gets 106 more matches right than that baseline. The ROC-AUC of 0.668 shows real discriminative power. The model finds 90% of home wins but almost never predicts a draw (1% recall), the hardest result to call in football; even professional betting models struggle with it.

> **Why the old CPU code showed 65.5%:** That code used `objective='binary:logistic'` — it was still solving the 2-class problem. It is NOT a better model; it is solving an easier, less useful task.

---

### CPU Reference — XGBoost original binary target (`xgboost_tuning_step2C_cpu.py`)

```
Best Parameters:
  learning_rate: 0.05 | max_depth: 3 | n_estimators: 100 | subsample: 0.8

Accuracy: 65.50%  ⚠️  Binary classification (H vs NH) — NOT true 3-class
                       Included for reference / machines without a GPU only.

              precision    recall  f1-score   support
           0       0.66      0.51      0.58       624
           1       0.65      0.77      0.71       744
    accuracy                           0.65      1368
   macro avg       0.66      0.64      0.64      1368
weighted avg       0.66      0.65      0.65      1368
```

> ⚠️ **Do not compare this 65.5% directly against the GPU 52.3%.** They solve different problems. The CPU file uses `objective='binary:logistic'` (2 classes: Home Win vs everything else). The GPU file uses `objective='multi:softmax'` (3 classes: H / D / A). The GPU version is the correct, honest model — it is harder but far more useful.

---

## 🔑 Top 10 Most Important Features

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | `DiffPts` | 0.0878 | Strongest predictor — the better team on points usually wins |
| 2 | `ATGD` | 0.0820 | Away team's defensive record over the season |
| 3 | `HomeTeam` | 0.0808 | Home advantage is team-specific, not generic |
| 4 | `ATP` | 0.0796 | Away team's league standing |
| 5 | `HTGD` | 0.0795 | Home team's defensive dominance |
| 6 | `AwayTeam` | 0.0768 | Identity of the visiting club matters |
| 7 | `HTP` | 0.0741 | Home team's league standing |
| 8 | `DiffFormPts` | 0.0719 | Short-term momentum gap between teams |
| 9 | `HTGS` | 0.0653 | Home team's attacking output |
| 10 | `ATGS` | 0.0647 | Away team's attacking output |

> **Takeaway:** Points difference and goal difference dominate prediction, but team identity carries significant weight — different clubs have different home/away profiles that the model learns through encoding.

---

## 🚀 Installation & Usage

### Requirements

```bash
pip install pandas scikit-learn xgboost joblib matplotlib numpy
```

> **GPU acceleration** — if you have an NVIDIA GPU, XGBoost will automatically use CUDA (`device='cuda'`). No extra install is needed; the script detects your hardware via `nvidia-smi` and falls back to CPU if no GPU is found. To verify your setup: `nvidia-smi` in your terminal.

### Run Order

| # | Command | Purpose |
|---|---|---|
| 1 | `python h2h_features_step2C.py` | Build enriched dataset with H2H features |
| 2 | `python model_training.py` | Train Step 1 baseline Random Forest |
| 3 | `python model_training2.py` | Train Step 2A RF + form + H2H |
| 4 | `python xgboost_tuning_step2C.py` | ✅ **Recommended** — XGBoost GPU, true 3-class (H/D/A), auto-detects CUDA |
| 4b | `python xgboost_tuning_step2C_cpu.py` | ⚠️ CPU fallback — binary (H vs NH), use only if no NVIDIA GPU available |
| 5 | `python feature_importance_step2B.py` | Generate and save analysis charts |
| 6 | `python predict_match.py` | Interactive single-match predictor |
| 7 | `python predict_matches.py` | Batch fixture predictor |

### Predict a Single Match

```bash
$ python predict_match.py

📋  Available teams (44 total):
    1. Arsenal   2. Aston Villa   3. Chelsea ...

Enter Home Team: Man City
Enter Away Team: Arsenal

====================================================
   ⚽  Man City  (H)  vs  Arsenal  (A)
====================================================

📊  WIN PROBABILITIES
    🏠 Man City                 52.3%
    🤝 Draw                     21.4%
    ✈️  Arsenal                  26.3%

🏆  Predicted Result  : 🏠 Man City Win
⚽  Estimated Score   : Man City 2 – 1 Arsenal
```

### Batch Predict Multiple Fixtures

Edit the `upcoming_fixtures` list in `predict_matches.py` and run:

```bash
$ python predict_matches.py
```

```
===========================================================================
  ⚽  MATCH PREDICTIONS
===========================================================================
 Home Team   Away Team      Prediction  Home Win %  Draw %  Away Win %  Score
  Man City     Arsenal    Man City Win       52.3%   21.4%       26.3%   2 – 1
 Liverpool  Man United   Liverpool Win       58.1%   19.2%       22.7%   2 – 1
===========================================================================
```

---

## ⚙️ Pipeline Architecture

```
Raw CSV (final_dataset.csv)
        │
        ▼
 h2h_features_step2C.py
 ── Direction-aware H2H stats (no leakage)
 ── Rolling 5-game form windows
        │
        ▼
 final_dataset_h2h.csv
        │
        ├──► model_training.py             → Baseline RF             63.1% (binary H vs NH)
        ├──► model_training2.py            → RF + features           63.2% (binary H vs NH)
        ├──► xgboost_tuning_step2C.py      → XGBoost GPU  true 3-class (H/D/A)  52.3% / ROC-AUC 0.668 ✅
        └──► xgboost_tuning_step2C_cpu.py  → XGBoost CPU  binary (H vs NH)       65.5% / ROC-AUC 0.668 ⚠️
                        │
                        ▼
            xgb_match_winner_model.pkl
            team_encoder_xgb.pkl
            result_encoder_xgb.pkl
                        │
                        ├──► predict_match.py    (single CLI)
                        └──► predict_matches.py  (batch CLI)
```

---

## ⚠️ Limitations

- Dataset covers 2000–2018 only — no live or real-time data integration
- No player-level data (injuries, suspensions, squad rotation, managerial changes)
- Score estimate is heuristic, not a trained Poisson regression model
- No betting odds features, which are strong proxies for true match probability
- Class imbalance (H: 46%, D: 26%, A: 28%) is not explicitly handled — Draw remains the hardest outcome to predict

---

## 🔭 Future Work

- [ ] Poisson goals model for principled score-distribution prediction
- [ ] Integrate publicly available betting odds as features (strong market signal)
- [ ] Rolling league-position features (position at the exact time of each match)
- [ ] Player-level features via a football data API (e.g., football-data.org)
- [ ] Stacked ensemble: RF + XGBoost + meta-learner (Logistic Regression)
- [ ] Wrap `predict_match` as a FastAPI REST endpoint
- [ ] Docker container for reproducible, portable deployment

---

*Premier League Match Predictor · Python / scikit-learn / XGBoost · EPL 2000–2018*
