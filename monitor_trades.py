from db import get_trades
import pandas as pd
from datetime import datetime, timedelta

def analyze_live_trades():
    """Analyze trades from live simulation"""
    print("📊 Live Trading Analysis")
    print("="*50)
    
    # Get all trades
    trades = get_trades()
    
    if not trades:
        print("❌ No trades found in database")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(trades)
    
    # Filter for live simulation trades
    live_trades = df[df.get('trade_type', '') == 'LIVE_SIMULATION']
    
    if live_trades.empty:
        print("❌ No live simulation trades found")
        print("💡 Run the live simulation first: py live_simulation.py")
        return
    
    print(f"📈 Found {len(live_trades)} live simulation trades")
    
    # Basic statistics
    print(f"\n📊 Trade Statistics:")
    print(f"   Total Trades: {len(live_trades)}")
    
    # Signal analysis
    buy_trades = live_trades[live_trades['signal'] == 'BUY']
    sell_trades = live_trades[live_trades['signal'] == 'SELL']
    
    print(f"   BUY signals: {len(buy_trades)}")
    print(f"   SELL signals: {len(sell_trades)}")
    
    # Price analysis
    if 'price' in live_trades.columns:
        avg_price = live_trades['price'].mean()
        min_price = live_trades['price'].min()
        max_price = live_trades['price'].max()
        print(f"   Average Price: ${avg_price:.2f}")
        print(f"   Price Range: ${min_price:.2f} - ${max_price:.2f}")
    
    # Time analysis
    if 'timestamp' in live_trades.columns:
        live_trades['timestamp'] = pd.to_datetime(live_trades['timestamp'])
        start_time = live_trades['timestamp'].min()
        end_time = live_trades['timestamp'].max()
        print(f"   Trading Period: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Recent trades
    print(f"\n🕐 Recent Trades (Last 10):")
    recent_trades = live_trades.sort_values('timestamp', ascending=False).head(10)
    
    for _, trade in recent_trades.iterrows():
        timestamp = pd.to_datetime(trade['timestamp']).strftime('%H:%M:%S')
        signal = trade['signal']
        quantity = trade.get('quantity', 'N/A')
        price = trade.get('price', 'N/A')
        if isinstance(price, (int, float)):
            price = f"${price:.2f}"
        
        print(f"   {timestamp} | {signal} {quantity} shares @ {price}")
    
    # Model performance analysis
    if 'prediction' in live_trades.columns:
        print(f"\n🤖 Model Performance:")
        correct_predictions = 0
        total_predictions = len(live_trades)
        
        for _, trade in live_trades.iterrows():
            prediction = trade['prediction']
            signal = trade['signal']
            
            # Simple accuracy check (in real trading, you'd compare with actual price movements)
            if prediction == 1 and signal == 'BUY':
                correct_predictions += 1
            elif prediction == 0 and signal == 'SELL':
                correct_predictions += 1
        
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        print(f"   Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    
    print(f"\n💡 Analysis Complete!")
    print(f"📊 Check your Alpaca dashboard for order status and P&L")

def check_account_status():
    """Check current account status"""
    try:
        from trader import api
        account = api.get_account()
        positions = api.list_positions()
        
        print(f"\n💰 Account Status:")
        print(f"   Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        
        if positions:
            print(f"\n📊 Current Positions:")
            for pos in positions:
                print(f"   {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")
        else:
            print(f"\n📊 No current positions")
            
    except Exception as e:
        print(f"❌ Error checking account: {e}")

if __name__ == "__main__":
    analyze_live_trades()
    check_account_status() 