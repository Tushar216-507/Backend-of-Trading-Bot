from predictor import get_prediction
from trader import place_order
from db import save_trade
import datetime

print("🚀 Starting automated trading system...")
print("="*50)

try:
    # Get prediction from the model
    print("📊 Getting market prediction...")
    features, prediction = get_prediction()
    
    # Generate trading signal
    signal = "BUY" if prediction == 1 else "SELL"
    quantity = 10
    price = features['close']
    stock = features['stock']
    
    print(f"📈 Analysis Results:")
    print(f"   Stock: {stock}")
    print(f"   Current Price: ${price:.2f}")
    print(f"   Model Prediction: {prediction} ({'BUY' if prediction == 1 else 'SELL'})")
    print(f"   Signal: {signal}")
    print(f"   Quantity: {quantity}")
    
    # Place the order
    print(f"💼 Placing {signal} order for {quantity} shares of {stock}...")
    order_id = place_order(stock, signal, quantity)
    print(f"✅ Order placed with ID: {order_id}")
    
    # Calculate simulated P&L (for demo purposes)
    pnl = 10 if signal == "BUY" else -5
    outcome = "profit" if pnl > 0 else "loss"
    
    # Prepare trade data for database
    trade_data = {
        "stock": stock,
        "signal": signal,
        "quantity": quantity,
        "price": price,
        "timestamp": datetime.datetime.now(),
        "pnl": pnl,
        "outcome": outcome,
        "order_id": order_id,
        "prediction_confidence": float(prediction)
    }
    
    # Save to database
    print("💾 Saving trade to database...")
    save_trade(trade_data)
    
    print("="*50)
    print("✅ Trade executed and logged successfully!")
    print(f"📊 Trade Summary:")
    print(f"   {signal} {quantity} shares of {stock} at ${price:.2f}")
    print(f"   Simulated P&L: ${pnl}")
    print(f"   Outcome: {outcome.upper()}")
    
except Exception as e:
    print(f"❌ Error in automated trading: {e}")
    print("🔧 Please check your API credentials and network connection")
