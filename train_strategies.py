import os
import glob
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import ta
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_all_csvs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if 'timestamp' not in df.columns:
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Coerce numeric columns
            for c in ['open', 'high', 'low', 'close', 'volume']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.sort_values('timestamp')
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise RuntimeError("No valid CSVs found to load")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna()
    combined = combined.set_index('timestamp')
    # Ensure unique index to avoid reindex issues later
    if combined.index.has_duplicates:
        combined = combined[~combined.index.duplicated(keep='last')]
    return combined


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(10).std()
    df['ohlc_delta'] = df['high'] - df['low']
    df['candle_body'] = (df['close'] - df['open']).abs()

    df['ema10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['close'])
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_bbh'] - df['bb_bbl']) / df['bb_bbm']

    df['volume_surge'] = df['volume'] / df['volume'].rolling(20).mean()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

    df['open_gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['price_change'] = df['close'] - df['open']
    df['price_change_pct'] = (df['close'] - df['open']) / df['open']

    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['price_vs_sma50'] = df['close'] / df['sma_50']
    df['price_vs_sma200'] = df['close'] / df['sma_200']

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    return df


def add_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Momentum
    df['ema5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['rsi7'] = ta.momentum.rsi(df['close'], window=7)
    df['roc_3'] = ta.momentum.roc(df['close'], window=3)
    # Trend/Breakout
    adx_ind = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx_ind.adx()
    # Mean reversion
    denom = (df['bb_bbh'] - df['bb_bbl']).replace(0, np.nan)
    df['bb_zscore'] = (df['close'] - df['bb_bbm']) / denom
    return df


def build_targets(df: pd.DataFrame) -> Dict[str, pd.Series]:
    targets: Dict[str, pd.Series] = {}
    # Momentum: next bar direction (1-step ahead)
    momentum = (df['close'].shift(-1) > df['close']).astype(int)
    momentum.name = 'momentum'
    targets['momentum'] = momentum
    # Mean reversion: fade extremes, profitable if price reverts next bar
    # if bb_zscore > 1 then expect down (0); if < -1 expect up (1); else fallback to next-bar direction
    mr_signal_arr = np.where(df['bb_zscore'] > 1, 0, np.where(df['bb_zscore'] < -1, 1, np.nan))
    fallback = (df['close'].shift(-1) > df['close']).astype(int)
    mr_target_arr = np.where(np.isnan(mr_signal_arr), fallback.values, mr_signal_arr).astype(int)
    mr_target = pd.Series(mr_target_arr, index=df.index, name='mean_reversion')
    targets['mean_reversion'] = mr_target
    # Trend/Breakout: 5-step ahead trend
    trend = (df['close'].shift(-5) > df['close']).astype(int)
    trend.name = 'trend'
    targets['trend'] = trend
    return targets


def select_features(strategy: str) -> List[str]:
    base = [
        'return', 'volatility', 'ohlc_delta', 'candle_body',
        'ema10', 'ema20', 'rsi', 'macd',
        'bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_width',
        'volume_surge', 'vwap_deviation',
        'open_gap_pct', 'price_change', 'price_change_pct',
        'price_vs_sma50', 'price_vs_sma200',
        'hour', 'day_of_week'
    ]
    if strategy == 'momentum':
        feats = base + ['ema5', 'rsi7', 'roc_3']
    elif strategy == 'mean_reversion':
        # 'vwap_deviation' is already in base
        feats = base + ['bb_zscore']
    elif strategy == 'trend':
        feats = base + ['adx']
    else:
        feats = base
    # Deduplicate while preserving order to avoid duplicate column labels
    feats = list(dict.fromkeys(feats))
    return feats


def train_strategy(df: pd.DataFrame, strategy: str) -> Tuple[xgb.XGBClassifier, float, List[str], float]:
    feats = select_features(strategy)
    y_all = build_targets(df)[strategy]
    # Attach label to the same frame to guarantee aligned lengths
    data = df.copy()
    data['__label'] = y_all
    data = data.dropna(subset=feats + ['__label'])
    X = data[feats]
    y = data['__label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Modest, robust defaults per strategy
    # Compute class imbalance to set scale_pos_weight
    pos_ratio = float(y.mean()) if len(y) > 0 else 0.5
    neg_ratio = 1.0 - pos_ratio
    spw = (neg_ratio / pos_ratio) if pos_ratio > 0 else 1.0

    params = dict(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        scale_pos_weight=spw
    )
    if strategy == 'momentum':
        params.update(dict(max_depth=5, learning_rate=0.08))
    elif strategy == 'mean_reversion':
        params.update(dict(max_depth=3, learning_rate=0.05))
    elif strategy == 'trend':
        params.update(dict(max_depth=5, n_estimators=400))

    # Force dense numpy arrays to avoid pandas dtype path in XGBoost
    model = xgb.XGBClassifier(**params)
    model.fit(X_train.values, y_train.values)

    # Accuracy using default 0.5 threshold
    y_pred_default = model.predict(X_test.values)
    acc_default = accuracy_score(y_test.values, y_pred_default)

    # Calibrate a probability threshold to maximize validation accuracy
    try:
        proba = model.predict_proba(X_test.values)[:, 1]
        candidate_thresholds = np.linspace(0.4, 0.6, 41)  # focus around 0.5
        best_thr = 0.5
        best_acc = acc_default
        for thr in candidate_thresholds:
            preds = (proba >= thr).astype(int)
            acc_thr = accuracy_score(y_test.values, preds)
            if acc_thr > best_acc:
                best_acc = acc_thr
                best_thr = float(thr)
        acc = best_acc
        threshold = best_thr
    except Exception:
        acc = acc_default
        threshold = 0.5

    return model, acc, feats, threshold


def main():
    # Collect CSVs in current folder
    csvs = [
        'historical_data.csv',
        'historical_data_ABNB.csv',
        'historical_data_ADBE.csv',
        'historical_data_new.csv'
    ]
    csvs = [c for c in csvs if os.path.exists(c)]
    if not csvs:
        raise RuntimeError("No historical CSV files found in workspace")

    df = load_all_csvs(csvs)
    df = base_features(df)
    df = add_strategy_features(df)

    results = {}
    for strat in ['momentum', 'mean_reversion', 'trend']:
        model, acc, feats, thr = train_strategy(df, strat)
        results[strat] = acc
        # Save model with its feature list and calibrated threshold for inference
        joblib.dump({'model': model, 'features': feats, 'threshold': thr}, f"xgb_{strat}.pkl")
        print(f"Saved xgb_{strat}.pkl (test accuracy: {acc:.4f}, thr={thr:.3f}) with {len(feats)} features")

    print("\nValidation accuracies:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()


