import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from db import get_trades
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def retrain_existing_model(n_last=100):
    print(f"🔄 Retraining 3-model ensemble using the last {n_last} trades...")
    trades = get_trades()
    if not trades or len(trades) < 10:
        print("❌ Not enough trades in database to retrain.")
        return

    trades = trades[-n_last:]
    df = pd.DataFrame(trades)

    features = [
        'return', 'volatility', 'ohlc_delta', 'candle_body',
        'ema10', 'ema20', 'rsi', 'macd',
        'bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_width',
        'volume_surge', 'vwap_deviation',
        'open_gap_pct', 'price_change', 'price_change_pct',
        'price_vs_sma50', 'price_vs_sma200',
        'hour', 'day_of_week'
    ]

    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"❌ Missing features in trade data: {missing}")
        return

    # Remove rows with missing or invalid predictions/features
    if 'prediction' in df.columns:
        df = df[df['prediction'].notnull() & np.isfinite(df['prediction'])]
    elif 'signal' in df.columns:
        df = df[df['signal'].notnull()]
    else:
        print("❌ No target column found in trades.")
        return

    # Remove rows with missing features
    df = df.dropna(subset=features)

    # Now extract y and X from the filtered DataFrame
    if 'prediction' in df.columns:
        y = df['prediction'].astype(int)
    else:
        y = (df['signal'].str.upper() == 'BUY').astype(int)

    X = df[features].astype(float)

    # Train an ensemble of 3 models with different seeds
    seeds = [42, 123, 999]
    accuracies = []
    for i, seed in enumerate(seeds, start=1):
        model_i = xgb.XGBClassifier(
            objective='binary:logistic',
            learning_rate=0.1,
            n_estimators=200,
            max_depth=4,
            random_state=seed,
            eval_metric='logloss'
        )
        model_i.fit(X, y)
        y_pred = model_i.predict(X)
        acc = accuracy_score(y, y_pred)
        accuracies.append(acc)
        joblib.dump(model_i, f"xgb_model_{i}.pkl")
        print(f"✅ Model {i} retrained from trades (acc={acc:.2f}) -> saved as xgb_model_{i}.pkl")
    if accuracies:
        print(f"Ensemble mean accuracy (on trades): {np.mean(accuracies):.4f}")

if __name__ == "__main__":
    retrain_existing_model(n_last=100)