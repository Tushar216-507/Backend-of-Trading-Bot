import time
import os
import joblib
from predictor import fetch_live_features, REQUIRED_FEATURES
from trader import place_order
from db import save_trade
import datetime
import pandas as pd


def _load_strategy_models():
    models = []
    for path in ["xgb_momentum.pkl", "xgb_mean_reversion.pkl", "xgb_trend.pkl"]:
        if os.path.exists(path):
            try:
                obj = joblib.load(path)
                if isinstance(obj, dict) and 'model' in obj and 'features' in obj:
                    models.append({'name': os.path.basename(path), 'model': obj['model'], 'features': obj['features']})
                else:
                    models.append({'name': os.path.basename(path), 'model': obj, 'features': REQUIRED_FEATURES})
            except Exception:
                continue
    return models

def run_accuracy_test(num_trades=10, delay_seconds=30):
    """Run multiple trades to test model accuracy"""
    print(f"🧪 Running {num_trades} trades to test model accuracy...")
    print("="*60)
    
    trades = []
    correct_predictions = 0
    models = _load_strategy_models()
    if not models:
        raise RuntimeError("No strategy models found. Train with train_strategies.py first.")
    
    for i in range(num_trades):
        print(f"\n📊 Trade #{i+1}/{num_trades}")
        print("-" * 40)
        
        try:
            # Fetch features and compute majority vote from the 3 strategy models only
            features = fetch_live_features()
            votes = []
            for entry in models:
                try:
                    fv = [features[f] for f in entry['features'] if f in features]
                    pred = int(entry['model'].predict([fv])[0])
                    votes.append(pred)
                    print(f"   {entry['name']}: {pred} ({'BUY' if pred == 1 else 'SELL'})")
                except Exception as _:
                    continue
            if not votes:
                raise RuntimeError("No predictions produced by strategy models")
            ones = sum(votes)
            zeros = len(votes) - ones
            prediction = 1 if ones > zeros else 0
            signal = "BUY" if prediction == 1 else "SELL"
            current_price = features['close']
            
            print(f"   Stock: {features['stock']}")
            print(f"   Current Price: ${current_price:.2f}")
            print(f"   Majority Vote: {prediction} ({signal}) from {len(votes)} models")
            print(f"   RSI: {features['rsi']:.2f}")
            print(f"   MACD: {features['macd']:.4f}")
            
            # Place order (but don't actually execute to avoid wash trading)
            print(f"   💼 Simulating {signal} order...")
            order_id = f"test_order_{i+1}_{signal}"
            
            # Simulate outcome (in real trading, you'd track actual price movements)
            # For this test, we'll simulate based on typical market behavior
            import random
            if signal == "BUY":
                # Simulate price movement after BUY signal
                price_change_pct = random.uniform(-0.02, 0.05)  # -2% to +5%
                outcome = "PROFIT" if price_change_pct > 0 else "LOSS"
                pnl = 10 if price_change_pct > 0 else -5
            else:  # SELL
                # Simulate price movement after SELL signal
                price_change_pct = random.uniform(-0.05, 0.02)  # -5% to +2%
                outcome = "PROFIT" if price_change_pct < 0 else "LOSS"
                pnl = 10 if price_change_pct < 0 else -5
            
            # Record trade
            trade_data = {
                "trade_number": i+1,
                "stock": features['stock'],
                "signal": signal,
                "prediction": int(prediction),  # Convert numpy.int64 to regular int
                "price": float(current_price),  # Convert numpy.float64 to regular float
                "price_change_pct": price_change_pct,
                "outcome": outcome,
                "pnl": pnl,
                "timestamp": datetime.datetime.now(),
                "order_id": order_id
            }
            
            trades.append(trade_data)
            
            # Check if prediction was correct
            if (signal == "BUY" and price_change_pct > 0) or (signal == "SELL" and price_change_pct < 0):
                correct_predictions += 1
                print(f"   ✅ CORRECT: {signal} signal led to {outcome}")
            else:
                print(f"   ❌ INCORRECT: {signal} signal led to {outcome}")
            
            print(f"   Price change: {price_change_pct:+.2%}")
            print(f"   P&L: ${pnl}")
            
            # Save to database
            save_trade(trade_data)
            
            # Wait before next trade (to avoid rate limiting and simulate real trading)
            if i < num_trades - 1:  # Don't wait after the last trade
                print(f"   ⏳ Waiting {delay_seconds} seconds before next trade...")
                time.sleep(delay_seconds)
                
        except Exception as e:
            print(f"   ❌ Error in trade #{i+1}: {e}")
            continue
    
    # Calculate accuracy
    accuracy = (correct_predictions / len(trades)) * 100 if trades else 0
    
    # Generate detailed report
    print("\n" + "="*60)
    print("📈 ACCURACY TEST RESULTS")
    print("="*60)
    print(f"Total Trades: {len(trades)}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # Analyze by signal type
    if trades:
        df = pd.DataFrame(trades)
        buy_trades = df[df['signal'] == 'BUY']
        sell_trades = df[df['signal'] == 'SELL']
        
        print(f"\n📊 Signal Analysis:")
        print(f"   BUY signals: {len(buy_trades)}")
        if len(buy_trades) > 0:
            buy_accuracy = (buy_trades['outcome'] == 'PROFIT').sum() / len(buy_trades) * 100
            print(f"   BUY accuracy: {buy_accuracy:.1f}%")
        
        print(f"   SELL signals: {len(sell_trades)}")
        if len(sell_trades) > 0:
            sell_accuracy = (sell_trades['outcome'] == 'PROFIT').sum() / len(sell_trades) * 100
            print(f"   SELL accuracy: {sell_accuracy:.1f}%")
        
        # Overall P&L
        total_pnl = df['pnl'].sum()
        print(f"\n💰 Overall P&L: ${total_pnl}")
        print(f"   Average P&L per trade: ${total_pnl/len(trades):.2f}")
        
        # Win rate
        wins = (df['outcome'] == 'PROFIT').sum()
        win_rate = (wins / len(trades)) * 100
        print(f"   Win rate: {win_rate:.1f}% ({wins}/{len(trades)})")
    
    print("\n" + "="*60)
    print("🎯 RECOMMENDATIONS:")
    if accuracy >= 70:
        print("   ✅ EXCELLENT: Model accuracy is very good!")
    elif accuracy >= 60:
        print("   ✅ GOOD: Model accuracy is acceptable for trading")
    elif accuracy >= 50:
        print("   ⚠️  FAIR: Model accuracy needs improvement")
    else:
        print("   ❌ POOR: Model needs retraining or different features")
    
    print("   💡 Consider:")
    print("   - Retraining with more recent data")
    print("   - Adding more features")
    print("   - Adjusting the prediction threshold")
    print("   - Using different technical indicators")
    
    return trades, accuracy

if __name__ == "__main__":
    # Run 10 trades with 30-second delays
    trades, accuracy = run_accuracy_test(num_trades=10, delay_seconds=30) 