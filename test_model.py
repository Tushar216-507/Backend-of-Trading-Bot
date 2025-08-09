import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib

from predictor import fetch_live_features, REQUIRED_FEATURES
from train_strategies import base_features, add_strategy_features, build_targets


def _load_models() -> List[Dict]:
    models: List[Dict] = []
    # Strategy models first
    for path in ["xgb_momentum.pkl", "xgb_mean_reversion.pkl", "xgb_trend.pkl"]:
        if os.path.exists(path):
            try:
                obj = joblib.load(path)
                if isinstance(obj, dict) and 'model' in obj and 'features' in obj:
                    models.append({
                        'name': os.path.splitext(os.path.basename(path))[0],
                        'model': obj['model'],
                        'features': obj['features']
                    })
                else:
                    models.append({
                        'name': os.path.splitext(os.path.basename(path))[0],
                        'model': obj,
                        'features': REQUIRED_FEATURES
                    })
            except Exception:
                pass
    # Legacy ensemble
    if not models:
        for path in ["xgb_model_1.pkl", "xgb_model_2.pkl", "xgb_model_3.pkl"]:
            if os.path.exists(path):
                try:
                    obj = joblib.load(path)
                    models.append({'name': os.path.basename(path), 'model': obj, 'features': REQUIRED_FEATURES})
                except Exception:
                    pass
    # Single fallback
    if not models and os.path.exists("xgb_model.pkl"):
        obj = joblib.load("xgb_model.pkl")
        models.append({'name': 'xgb_model', 'model': obj, 'features': REQUIRED_FEATURES})
    return models


def test_live_predictions() -> None:
    print("🧪 Live predictions from all available models")
    print("=" * 60)
    models = _load_models()
    if not models:
        print("❌ No models found in workspace")
        return
    features = fetch_live_features()
    votes = []
    for entry in models:
        fv = [features[f] for f in entry['features'] if f in features]
        pred = int(entry['model'].predict([fv])[0])
        votes.append(pred)
        print(f"- {entry['name']}: {pred} ({'BUY' if pred == 1 else 'SELL'})")
    ones = sum(votes)
    zeros = len(votes) - ones
    maj = 1 if ones > zeros else 0
    print(f"Majority vote: {maj} ({'BUY' if maj == 1 else 'SELL'}) from {len(votes)} models")


def _load_all_csvs() -> pd.DataFrame:
    files = [
        'historical_data.csv',
        'historical_data_ABNB.csv',
        'historical_data_ADBE.csv',
        'historical_data_new.csv'
    ]
    frames = []
    for p in files:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
            if 'timestamp' not in df.columns:
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise RuntimeError("No CSVs found for accuracy test")
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna()
    all_df = all_df.sort_values('timestamp').set_index('timestamp')
    if all_df.index.has_duplicates:
        all_df = all_df[~all_df.index.duplicated(keep='last')]
    return all_df.tail(20000)  # cap for speed


def evaluate_historical_accuracy() -> None:
    print("\n🧪 Historical accuracy check for each model")
    print("=" * 60)
    models = _load_models()
    if not models:
        print("❌ No models found to evaluate")
        return
    df = _load_all_csvs()
    df = base_features(df)
    df = add_strategy_features(df)
    targets = build_targets(df)

    for entry in models:
        name = entry['name']
        feats = entry['features']
        # Strategy name inference
        if 'momentum' in name:
            target = targets['momentum']
        elif 'mean_reversion' in name:
            target = targets['mean_reversion']
        elif 'trend' in name:
            target = targets['trend']
        else:
            target = targets['momentum']  # default

        data = df.copy()
        data['__y'] = target
        needed_cols = [f for f in feats if f in data.columns]
        data = data.dropna(subset=needed_cols + ['__y'])
        if len(data) < 100:
            print(f"- {name}: not enough data after preprocessing to evaluate")
            continue
        X = data[needed_cols].values
        y = data['__y'].astype(int).values
        # Simple time-based split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        # Fit a copy (do not modify original) — or skip fitting and only predict if model supports
        try:
            preds = entry['model'].predict(X_test)
        except Exception:
            # As a fallback, refit on train to ensure estimator is ready
            entry['model'].fit(X_train, y_train)
            preds = entry['model'].predict(X_test)
        acc = (preds == y_test).mean()
        pred_buys = int((preds == 1).sum())
        pred_sells = int((preds == 0).sum())
        actual_buys = int((y_test == 1).sum())
        actual_sells = int((y_test == 0).sum())
        print(f"- {name}: accuracy={acc:.4f} on {len(y_test)} samples | BUY/S ell preds={pred_buys}/{pred_sells} | BUY/S ell actual={actual_buys}/{actual_sells}")


if __name__ == "__main__":
    test_live_predictions()
    evaluate_historical_accuracy()