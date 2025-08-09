import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import ta
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from typing import List

def load_and_prepare_data(file_path):
    """Load and prepare new data"""
    print("Loading new data...")
    df = pd.read_csv(file_path)
    
    # Convert columns to proper data types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    # Set index
    df = df.set_index('timestamp')
    
    # Get last 30 days of data
    last_date = df.index.max()
    start_date = last_date - timedelta(days=30)
    df = df[df.index >= start_date]
    
    print(f"Selected data from {start_date.date()} to {last_date.date()}")
    return df

def create_features(df):
    """Create features for model training"""
    # Price-Based Features
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(10).std()
    df['ohlc_delta'] = df['high'] - df['low']
    df['candle_body'] = abs(df['close'] - df['open'])

    # Technical Indicators
    df['ema10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['close'])
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_bbh'] - df['bb_bbl']) / df['bb_bbm']

    # Volume Features
    df['volume_surge'] = df['volume'] / df['volume'].rolling(20).mean()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

    # Market Context Features
    df['open_gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['price_change'] = df['close'] - df['open']
    df['price_change_pct'] = (df['close'] - df['open']) / df['open']

    # Moving averages
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['price_vs_sma50'] = df['close'] / df['sma_50']
    df['price_vs_sma200'] = df['close'] / df['sma_200']

    # Time Features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    return df

def retrain_model_with_new_data():
    """Main function to retrain the model with new data"""
    try:
        # Loading existing models is optional; we will retrain fresh ensemble
        print("Preparing data for retraining...")
        
        # Load and prepare new data
        df = load_and_prepare_data("historical_data_ADBE.csv")
        df = create_features(df)
        
        # Create target variable
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        df = df[:-1]  # Remove last row since it won't have a target
        
        # Select features
        features = [
            'return', 'volatility', 'ohlc_delta', 'candle_body',
            'ema10', 'ema20', 'rsi', 'macd',
            'bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_width',
            'volume_surge', 'vwap_deviation',
            'open_gap_pct', 'price_change', 'price_change_pct',
            'price_vs_sma50', 'price_vs_sma200',
            'hour', 'day_of_week'
        ]
        
        # Prepare data for training
        X = df[features].dropna()
        y = df['target'].loc[X.index]
        
        # Configure base parameters
        base_params = dict(
            objective='binary:logistic',
            base_score=0.5,
            scale_pos_weight=1,
            learning_rate=0.1,
            n_estimators=200,
            max_depth=4,
            eval_metric='logloss'
        )

        print("Retraining 3-model ensemble with new data...")
        class_weights = dict(zip(
            np.unique(y),
            1 / np.bincount(y) * len(y) / 2.0
        ))
        sample_weights = np.array([class_weights[yi] for yi in y])

        accuracies: List[float] = []
        for i, seed in enumerate([42, 123, 999], start=1):
            model_i = xgb.XGBClassifier(random_state=seed, **base_params)
            model_i.fit(X, y, sample_weight=sample_weights, verbose=True)
            y_pred = model_i.predict(X)
            acc = accuracy_score(y, y_pred)
            accuracies.append(acc)
            joblib.dump(model_i, f"xgb_model_{i}.pkl")
            print(f"Model {i} accuracy on new data: {acc:.4f} -> saved as xgb_model_{i}.pkl")

        print(f"Ensemble training complete. Mean accuracy: {np.mean(accuracies):.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error during retraining: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting model retraining process...")
    success = retrain_model_with_new_data()
    if success:
        print("✅ Model retraining completed successfully!")
    else:
        print("❌ Model retraining failed!")