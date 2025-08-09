import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import ta
from typing import List

print("Loading stock data...")
# Use local file relative to workspace
df = pd.read_csv("historical_data_ABNB.csv")

# Data cleaning and preprocessing
df = df.dropna(subset=['timestamp'])
df = df[df['timestamp'] != 'NaN']

# Convert columns to proper data types
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['open'] = pd.to_numeric(df['open'], errors='coerce')
df['high'] = pd.to_numeric(df['high'], errors='coerce')
df['low'] = pd.to_numeric(df['low'], errors='coerce')
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

# Remove any rows with NaN values
df = df.dropna()
df.set_index('timestamp', inplace=True)
print(f"Data loaded successfully. Shape: {df.shape}")

# Enhanced Feature Engineering
def create_features(df):
    # Price-Based Features
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(10).std()
    df['ohlc_delta'] = df['high'] - df['low']
    df['candle_body'] = abs(df['close'] - df['open'])

    # Enhanced Technical Indicators
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

# Create features
df = create_features(df)

# Create target variable
df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
df = df[:-1]  # Remove last row since it won't have a target

# Select features for model
features: List[str] = [
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Grid search for hyperparameter optimization
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

model = xgb.XGBClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Use best params to train an ensemble of 3 models with different seeds
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

accuracies = []
for i, seed in enumerate([42, 123, 999], start=1):
    model_i = xgb.XGBClassifier(random_state=seed, **best_params)
    model_i.fit(X_train, y_train)
    y_pred = model_i.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    joblib.dump(model_i, f"xgb_model_{i}.pkl")
    print(f"Model {i} (seed={seed}) accuracy: {acc:.4f} -> saved as xgb_model_{i}.pkl")

print(f"Ensemble mean accuracy: {np.mean(accuracies):.4f}")