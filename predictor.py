import pandas as pd
import numpy as np
import ta
import joblib
from alpaca_trade_api.rest import REST
import os
from typing import List, Tuple, Dict
from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, ALPACA_BASE_URL

# Use config file for Alpaca API credentials
api = REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, base_url=ALPACA_BASE_URL)

# Keep a single authoritative list for feature ordering
REQUIRED_FEATURES: List[str] = [
    'return', 'volatility', 'ohlc_delta', 'candle_body',
    'ema10', 'ema20', 'rsi', 'macd',
    'bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_width',
    'volume_surge', 'vwap_deviation',
    'open_gap_pct', 'price_change', 'price_change_pct',
    'price_vs_sma50', 'price_vs_sma200',
    'hour', 'day_of_week'
]

# Extra features for strategy-specific models
STRATEGY_EXTRA_FEATURES: List[str] = [
    'ema5',        # momentum
    'rsi7',        # momentum
    'roc_3',       # momentum
    'bb_zscore',   # mean reversion
    'adx'          # trend/breakout
]

def fetch_live_features():
    symbol = "AAPL"
    try:
        # Fetch more data to calculate moving averages properly
        print(f"Fetching live data for {symbol}...")
        barset = api.get_bars(symbol, timeframe="1Min", limit=250)
        
        # Handle different API response formats
        if hasattr(barset, 'df'):
            df = barset.df
        else:
            # Convert to DataFrame if it's not already
            df = pd.DataFrame([bar._raw for bar in barset])
        
        # Handle symbol column filtering if it exists
        if 'symbol' in df.columns:
            df = df[df['symbol'] == symbol].copy()
        
        # Reset index and ensure we have the required columns
        if df.index.name != 'timestamp' and 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        elif 't' in df.columns:  # Some API versions use 't' for timestamp
            df.set_index('t', inplace=True)
            df.index.name = 'timestamp'
        
        # Rename columns if needed (some API versions use different names)
        column_mapping = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        df.rename(columns=column_mapping, inplace=True)
        
        if df.empty:
            raise ValueError("No data received from API")
        
        print(f"Retrieved {len(df)} data points for {symbol}")
        
        # Calculate features exactly as in training (same names and windows)
        df['return'] = df['close'].pct_change()
        df['volatility'] = df['return'].rolling(10).std()
        df['ohlc_delta'] = df['high'] - df['low']
        df['candle_body'] = abs(df['close'] - df['open'])
        
        # Technical Indicators (exact same as training)
        df['ema10'] = ta.trend.ema_indicator(df['close'], window=10)
        df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['close'])
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_bbm'] = bb.bollinger_mavg()
        df['bb_bbh'] = bb.bollinger_hband()
        df['bb_bbl'] = bb.bollinger_lband()
        
        # Volume Features
        df['volume_surge'] = df['volume'] / df['volume'].rolling(20).mean()
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = df['close'] - df['vwap']
        
        # Market Context Features for US stocks
        df['open_gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages for trend analysis (use shorter windows for limited data)
        available_data = len(df)
        sma_50_window = min(50, max(10, available_data // 4))
        sma_200_window = min(200, max(20, available_data // 2))
        
        df['sma_50'] = df['close'].rolling(sma_50_window).mean()
        df['sma_200'] = df['close'].rolling(sma_200_window).mean()
        df['price_vs_sma50'] = df['close'] / df['sma_50']
        df['price_vs_sma200'] = df['close'] / df['sma_200']
        
        print(f"Using SMA windows: {sma_50_window} and {sma_200_window} for {available_data} data points")
        
        # Time Features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # Extra features for strategy models (computed before dropna)
        try:
            df['ema5'] = ta.trend.ema_indicator(df['close'], window=5)
            df['rsi7'] = ta.momentum.rsi(df['close'], window=7)
            df['roc_3'] = ta.momentum.roc(df['close'], window=3)
            adx_ind = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['adx'] = adx_ind.adx()
            # Bollinger z-score relative to band width
            denom = (df['bb_bbh'] - df['bb_bbl']).replace(0, np.nan)
            df['bb_zscore'] = (df['close'] - df['bb_bbm']) / denom
        except Exception:
            pass
        
        # Drop NaN rows
        df.dropna(inplace=True)
        
        if df.empty:
            raise ValueError("No valid data after feature calculation")
        
        # Add new features
        df['bb_width'] = (df['bb_bbh'] - df['bb_bbl']) / df['bb_bbm']
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Get the latest row features (exact same order as training)
        latest_row = df.iloc[-1]
        latest_features = latest_row[REQUIRED_FEATURES].to_dict()
        # Attach strategy extras if available
        for f in STRATEGY_EXTRA_FEATURES:
            if f in latest_row.index:
                latest_features[f] = latest_row[f]
        
        return {**latest_features, 'stock': symbol, 'close': df['close'].iloc[-1]}
        
    except Exception as e:
        print(f"Error fetching live data: {e}")
        print("Using sample features for demo...")
        # Return sample features that match the training format
        return {
            'return': 0.001, 'volatility': 0.02, 'ohlc_delta': 0.5, 'candle_body': 0.3,
            'ema10': 150.0, 'ema20': 149.0, 'rsi': 55.0, 'macd': 0.1,
            'bb_bbm': 150.0, 'bb_bbh': 152.0, 'bb_bbl': 148.0,
            'volume_surge': 1.2, 'vwap_deviation': 0.1,
            'open_gap_pct': 0.001, 'price_change': 0.2, 'price_change_pct': 0.001,
            'price_vs_sma50': 1.01, 'price_vs_sma200': 1.05,
            'hour': 14, 'day_of_week': 2,
            # strategy extras (neutral defaults)
            'ema5': 150.0, 'rsi7': 55.0, 'roc_3': 0.001,
            'bb_zscore': 0.0, 'adx': 20.0,
            'stock': 'AAPL', 'close': 150.25
        }

def _load_ensemble_models() -> List[Dict]:
    """Load strategy models if present; otherwise legacy ensemble; fallback to single model.
    Returns list of dicts: { 'model': model_obj, 'features': feature_list }
    """
    loaded: List[Dict] = []

    # Preferred: strategy-specific models
    for path in ["xgb_momentum.pkl", "xgb_mean_reversion.pkl", "xgb_trend.pkl"]:
        if os.path.exists(path):
            try:
                obj = joblib.load(path)
                if isinstance(obj, dict) and 'model' in obj and 'features' in obj:
                    entry = {'model': obj['model'], 'features': obj['features']}
                    if 'threshold' in obj:
                        entry['threshold'] = obj['threshold']
                    loaded.append(entry)
                else:
                    loaded.append({'model': obj, 'features': REQUIRED_FEATURES, 'threshold': 0.5})
            except Exception:
                pass

    # Legacy ensemble
    if not loaded:
        for path in ["xgb_model_1.pkl", "xgb_model_2.pkl", "xgb_model_3.pkl"]:
            if os.path.exists(path):
                try:
                    obj = joblib.load(path)
                    loaded.append({'model': obj, 'features': REQUIRED_FEATURES})
                except Exception:
                    pass

    # Fallback single model
    if not loaded and os.path.exists("xgb_model.pkl"):
        obj = joblib.load("xgb_model.pkl")
        loaded.append({'model': obj, 'features': REQUIRED_FEATURES, 'threshold': 0.5})

    return loaded

def get_prediction() -> Tuple[Dict, int]:
    model_entries = _load_ensemble_models()
    features = fetch_live_features()

    # If only one model, use its declared feature list with threshold/proba decision
    if len(model_entries) == 1:
        entry = model_entries[0]
        fv = [features[f] for f in entry['features']]
        # Prefer probability-based decision
        try:
            proba = float(entry['model'].predict_proba([fv])[0][1])
            thr = float(entry.get('threshold', 0.5))
            confidence = abs(proba - thr)
            # HOLD if confidence is too low
            if confidence < 0.05:
                features['model_confidence'] = confidence
                return features, -1
            pred = 1 if proba >= thr else 0
        except Exception:
            pred = int(entry['model'].predict([fv])[0])
            features['model_confidence'] = 0.0
        # Post-decision risk filters
        rsi_val = features.get('rsi', 50)
        macd_val = features.get('macd', 0)
        p_vs_sma50 = features.get('price_vs_sma50', 1)
        if pred == 1 and not (p_vs_sma50 > 1.0 and macd_val > 0 and rsi_val < 70):
            return features, -1
        if pred == 0 and not (p_vs_sma50 < 1.0 and macd_val < 0 and rsi_val > 30):
            return features, -1
        return features, pred

    # Majority vote among models with their own feature sets
    votes: List[int] = []
    margins: List[float] = []
    for entry in model_entries:
        try:
            fv = [features[f] for f in entry['features']]
            # probability vote with model-specific threshold
            try:
                proba = float(entry['model'].predict_proba([fv])[0][1])
                thr = float(entry.get('threshold', 0.5))
                votes.append(1 if proba >= thr else 0)
                margins.append(proba - thr)
            except Exception:
                votes.append(int(entry['model'].predict([fv])[0]))
        except Exception:
            continue
    if not votes:
        return features, 0
    ones = sum(1 for v in votes if v == 1)
    zeros = len(votes) - ones
    if ones == zeros:
        features['model_confidence'] = 0.0
        return features, -1
    prediction = 1 if ones > zeros else 0
    avg_conf = float(np.mean([abs(m) for m in margins])) if margins else 0.0
    features['model_confidence'] = avg_conf
    if avg_conf < 0.05:
        return features, -1
    
    # Simple risk filters to reduce bad BUYs:
    # - If RSI is very high (>75) and model says BUY, downgrade to HOLD/SELL
    # - If ADX is low (<15) and model says TREND BUY, be cautious
    rsi_val = features.get('rsi', 50)
    adx_val = features.get('adx', 20)
    macd_val = features.get('macd', 0)
    p_vs_sma50 = features.get('price_vs_sma50', 1)
    if prediction == 1 and rsi_val > 75:
        return features, -1
    if prediction == 1 and adx_val < 15:
        return features, -1
    # Trend-aligned gating
    if prediction == 1 and not (p_vs_sma50 > 1.0 and macd_val > 0 and rsi_val < 70):
        return features, -1
    if prediction == 0 and not (p_vs_sma50 < 1.0 and macd_val < 0 and rsi_val > 30):
        return features, -1
    return features, prediction
