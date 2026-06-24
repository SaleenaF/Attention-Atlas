"""
Attention Atlas - Machine Learning Engine
Trains a Random Forest Classifier on historical video data to predict
performance tiers and save the model file for the dashboard optimizer.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Step up out of src/
DB_PATH = os.path.join(BASE_DIR, "attention_atlas.db")
MODEL_OUTPUT = os.path.join(BASE_DIR, "view_predictor.joblib")

def train_predictive_engine():
    print("🛰️ Booting Attention Atlas Machine Learning Engine...")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database '{DB_PATH}' not found. Please run pipeline.py first!")
        return

    # 1. Fetch data from SQLite
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM videos", conn)
    conn.close()

    if df.empty:
        print("❌ Error: The 'videos' table is empty. Check your pipeline script execution.")
        return

    # Drop any null values in critical columns
    df = df.dropna(subset=["views", "duration", "publish_month", "publish_dow", "format", "category"])

    # 2. Build Categorical Target Tiers using real percentile boundaries
    print("📈 Feature engineering historic performance milestones...")
    def calculate_performance_tier(views):
        if views <= 300:
            return "Low"
        elif views <= 1225:
            return "Mid"
        elif views <= 3200:
            return "High"
        else:
            return "Breakout"

    df["performance_tier"] = df["views"].apply(calculate_performance_tier)

    # Map model features to match your dashboard input parameters
    X = df[["duration", "publish_month", "format", "category", "publish_dow"]].copy()
    X.columns = ["duration_sec", "publish_month", "format", "category", "publish_dow"]
    y = df["performance_tier"]

    print(f"📊 Training distribution: {dict(y.value_counts())}")

    # 3. Construct the Machine Learning Pipeline Layout
    categorical_features = ["format", "category", "publish_dow"]
    numeric_features = ["duration_sec", "publish_month"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough" # Keeps numeric features unchanged
    )

    # Core Random Forest model structure
    model_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            class_weight="balanced" # Prevents data bias towards any single tier
        ))
    ])

    # 4. Train and Validate the Model
    print("🧠 Initializing random forest parameter sweeps...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_pipeline.fit(X_train, y_train)
    accuracy = model_pipeline.score(X_test, y_test)
    print(f"   🎯 Model training validation accuracy: {accuracy * 100:.2f}%")

    # 5. Export finalized artifacts
    joblib.dump(model_pipeline, MODEL_OUTPUT)
    print(f"💾 Success! Machine learning model saved as a deployment asset at:\n   -> '{MODEL_OUTPUT}'")

if __name__ == "__main__":
    train_predictive_engine()