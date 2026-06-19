import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os

print("Loading data...")
PROCESSED = r"C:\Users\HP\OneDrive\Desktop\bbl-cricket-dashboard\data\processed"

model_df = pd.read_parquet(os.path.join(PROCESSED, "model_df.parquet"))

FEATURES = [
    "cum_total_runs", "cum_wickets", "balls_remaining",
    "current_rr", "runs_needed", "required_rr", "rr_pressure", "target"
]
TARGET = "chaser_won"

# Train/test split by match
match_ids = model_df["match_id"].unique()
np.random.seed(42)
np.random.shuffle(match_ids)

split_point = int(len(match_ids) * 0.8)
train_ids   = match_ids[:split_point]

train_df = model_df[model_df["match_id"].isin(train_ids)]
X_train  = train_df[FEATURES]
y_train  = train_df[TARGET]

print("Training model...")
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
model_path = r"C:\Users\HP\OneDrive\Desktop\bbl-cricket-dashboard\app\model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("✅ Model retrained and saved successfully!")