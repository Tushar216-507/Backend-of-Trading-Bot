import joblib
import numpy as np

def test_sell_signal():
    """Test if the model can generate SELL signals with bearish market conditions"""
    print("🧪 Testing SELL signal generation...")
    print("="*50)
    
    # Load the model
    model = joblib.load("xgb_model.pkl")
    
    # Create bearish market conditions (features that typically indicate selling)
    bearish_features = {
        'return': -0.02,           # Negative return
        'volatility': 0.05,        # High volatility
        'ohlc_delta': 2.0,         # Large price range
        'candle_body': 1.5,        # Large candle body
        'ema10': 200.0,            # EMA10 below EMA20
        'ema20': 205.0,            # EMA20 above EMA10 (bearish crossover)
        'rsi': 75.0,               # Overbought RSI
        'macd': -0.5,              # Negative MACD
        'bb_bbm': 200.0,           # Bollinger Band middle
        'bb_bbh': 205.0,           # Bollinger Band high
        'bb_bbl': 195.0,           # Bollinger Band low
        'bb_width': 0.05,          # Add missing Bollinger Band width feature
        'volume_surge': 2.0,       # High volume
        'vwap_deviation': -2.0,    # Price below VWAP
        'open_gap_pct': -0.01,     # Gap down
        'price_change': -1.0,      # Negative price change
        'price_change_pct': -0.005, # Negative percentage change
        'price_vs_sma50': 0.95,    # Price below SMA50
        'price_vs_sma200': 0.90,   # Price below SMA200
        'hour': 14,                # Market hours
        'day_of_week': 2           # Tuesday
    }
    
    # Get feature values in the correct order
    feature_names = [
        'return', 'volatility', 'ohlc_delta', 'candle_body',
        'ema10', 'ema20', 'rsi', 'macd',
        'bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_width',
        'volume_surge', 'vwap_deviation',
        'open_gap_pct', 'price_change', 'price_change_pct',
        'price_vs_sma50', 'price_vs_sma200',
        'hour', 'day_of_week'
    ]
    
    feature_values = [bearish_features[name] for name in feature_names]
    
    # Get prediction
    prediction = model.predict([feature_values])[0]
    probabilities = model.predict_proba([feature_values])[0]
    
    print("📊 Bearish Market Simulation:")
    print(f"   RSI: {bearish_features['rsi']:.1f} (Overbought)")
    print(f"   MACD: {bearish_features['macd']:.3f} (Negative)")
    print(f"   Price vs SMA50: {bearish_features['price_vs_sma50']:.2f} (Below)")
    print(f"   Price vs SMA200: {bearish_features['price_vs_sma200']:.2f} (Below)")
    print(f"   Return: {bearish_features['return']:.3f} (Negative)")
    
    print(f"\n🎯 Model Prediction:")
    print(f"   Signal: {prediction} ({'BUY' if prediction == 1 else 'SELL'})")
    print(f"   BUY probability: {probabilities[1]:.3f} ({probabilities[1]*100:.1f}%)")
    print(f"   SELL probability: {probabilities[0]:.3f} ({probabilities[0]*100:.1f}%)")
    
    if prediction == 0:
        print("   ✅ SUCCESS: Model generated SELL signal!")
    else:
        print("   ⚠️  Model still generated BUY signal despite bearish conditions")
        print("   This might indicate the model needs retraining or different features")

if __name__ == "__main__":
    test_sell_signal()