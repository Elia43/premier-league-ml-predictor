import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the model and data
model_path = "data/match_winner_model.pkl"  # or match_winner_model_step2A.pkl if that’s your latest
data_path = "data/final_dataset.csv"

model = joblib.load(model_path)
data = pd.read_csv(data_path)

# Try to get feature names safely
try:
    features = model.feature_names_in_
except AttributeError:
    features = data.columns.drop(["FTR"], errors="ignore").tolist()

# Get importances (for RandomForest or similar models)
importances = getattr(model, "feature_importances_", None)

if importances is None:
    print("⚠️ Model has no feature_importances_. It might be a different type (like LogisticRegression).")
else:
    # Fix any mismatch in lengths automatically
    min_len = min(len(features), len(importances))
    features = features[:min_len]
    importances = importances[:min_len]

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
    plt.gca().invert_yaxis()
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.show()
