import pandas as pd
from datetime import datetime, timedelta
from db import get_trades
from trader import api
import pytz

def calculate_pnl_from_trades():
    """Calculate P&L from actual trades in Alpaca"""
    print("💰 Calculating Real P&L from Alpaca Trades")
    print("="*60)
    
    try:
        # Get account status
        account = api.get_account()
        current_portfolio_value = float(account.portfolio_value)
        current_cash = float(account.cash)
        
        # Get all positions
        positions = api.list_positions()
        
        # Get today's orders
        eastern = pytz.timezone('US/Eastern')
        today = datetime.now(eastern).date()
        
        orders = api.list_orders(
            status='all',
            after=today.strftime('%Y-%m-%d'),
            limit=500
        )
        
        print(f"📊 Account Summary:")
        print(f"   Portfolio Value: ${current_portfolio_value:,.2f}")
        print(f"   Cash: ${current_cash:,.2f}")
        print(f"   Total Value: ${current_portfolio_value + current_cash:,.2f}")
        
        # Calculate P&L from positions
        total_pnl = 0
        if positions:
            print(f"\n📈 Current Positions:")
            for pos in positions:
                symbol = pos.symbol
                qty = float(pos.qty)
                avg_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                market_value = float(pos.market_value)
                unrealized_pl = float(pos.unrealized_pl)
                
                total_pnl += unrealized_pl
                
                print(f"   {symbol}: {qty} shares @ ${avg_price:.2f}")
                print(f"     Current Price: ${current_price:.2f}")
                print(f"     Market Value: ${market_value:,.2f}")
                print(f"     Unrealized P&L: ${unrealized_pl:,.2f}")
        else:
            print(f"\n📈 No current positions")
        
        # Calculate P&L from completed orders
        if orders:
            print(f"\n📋 Today's Orders:")
            buy_orders = []
            sell_orders = []
            
            for order in orders:
                if order.status == 'filled':
                    if order.side == 'buy':
                        buy_orders.append({
                            'symbol': order.symbol,
                            'qty': float(order.qty),
                            'price': float(order.filled_avg_price),
                            'value': float(order.qty) * float(order.filled_avg_price)
                        })
                    elif order.side == 'sell':
                        sell_orders.append({
                            'symbol': order.symbol,
                            'qty': float(order.qty),
                            'price': float(order.filled_avg_price),
                            'value': float(order.qty) * float(order.filled_avg_price)
                        })
            
            # Calculate realized P&L
            realized_pnl = 0
            for sell in sell_orders:
                # Find corresponding buy orders
                symbol = sell['symbol']
                sell_qty = sell['qty']
                sell_price = sell['price']
                
                # Find matching buy orders for this symbol
                matching_buys = [b for b in buy_orders if b['symbol'] == symbol]
                
                for buy in matching_buys:
                    if sell_qty > 0 and buy['qty'] > 0:
                        trade_qty = min(sell_qty, buy['qty'])
                        trade_pnl = (sell_price - buy['price']) * trade_qty
                        realized_pnl += trade_pnl
                        
                        sell_qty -= trade_qty
                        buy['qty'] -= trade_qty
                        
                        if sell_qty <= 0:
                            break
            
            print(f"   Realized P&L: ${realized_pnl:,.2f}")
            print(f"   Unrealized P&L: ${total_pnl:,.2f}")
            print(f"   Total P&L: ${realized_pnl + total_pnl:,.2f}")
        
        return realized_pnl + total_pnl
        
    except Exception as e:
        print(f"❌ Error calculating P&L: {e}")
        return 0

def analyze_model_accuracy():
    """Analyze model accuracy from database trades"""
    print(f"\n🤖 Model Accuracy Analysis")
    print("="*60)
    
    # Get trades from database
    trades = get_trades()
    
    if not trades:
        print("❌ No trades found in database")
        return
    
    df = pd.DataFrame(trades)
    
    # Filter for today's live simulation trades
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern).date()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    today_trades = df[df['timestamp'].dt.date == today]
    
    if today_trades.empty:
        print("❌ No trades found for today")
        return
    
    print(f"📊 Today's Trading Summary:")
    print(f"   Total Trades: {len(today_trades)}")
    
    # Signal analysis
    buy_trades = today_trades[today_trades['signal'] == 'BUY']
    sell_trades = today_trades[today_trades['signal'] == 'SELL']
    
    print(f"   BUY signals: {len(buy_trades)}")
    print(f"   SELL signals: {len(sell_trades)}")
    
    # Price analysis
    if 'price' in today_trades.columns:
        avg_price = today_trades['price'].mean()
        min_price = today_trades['price'].min()
        max_price = today_trades['price'].max()
        print(f"   Average Price: ${avg_price:.2f}")
        print(f"   Price Range: ${min_price:.2f} - ${max_price:.2f}")
    
    # Model accuracy (based on predictions vs actual market movement)
    if 'prediction' in today_trades.columns:
        print(f"\n🎯 Model Performance:")
        
        # Get current AAPL price for comparison
        try:
            current_aapl = api.get_latest_trade('AAPL')
            current_price = float(current_aapl.price)
            print(f"   Current AAPL Price: ${current_price:.2f}")
            
            # Calculate accuracy based on price movement
            correct_predictions = 0
            total_predictions = len(today_trades)
            
            for _, trade in today_trades.iterrows():
                trade_price = trade['price']
                prediction = trade['prediction']
                signal = trade['signal']
                
                # Simple accuracy check: if price went up after BUY, or down after SELL
                if signal == 'BUY' and current_price > trade_price:
                    correct_predictions += 1
                elif signal == 'SELL' and current_price < trade_price:
                    correct_predictions += 1
            
            accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            print(f"   Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
            
        except Exception as e:
            print(f"   Could not calculate accuracy: {e}")

def show_detailed_trades():
    """Show detailed trade information"""
    print(f"\n📋 Detailed Trade Log")
    print("="*60)
    
    trades = get_trades()
    if not trades:
        print("❌ No trades found")
        return
    
    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Show today's trades
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern).date()
    today_trades = df[df['timestamp'].dt.date == today]
    
    if today_trades.empty:
        print("❌ No trades found for today")
        return
    
    print(f"🕐 Trade Details (Today):")
    for _, trade in today_trades.iterrows():
        timestamp = trade['timestamp'].strftime('%H:%M:%S')
        signal = trade['signal']
        quantity = trade.get('quantity', 'N/A')
        price = trade.get('price', 'N/A')
        prediction = trade.get('prediction', 'N/A')
        
        if isinstance(price, (int, float)):
            price = f"${price:.2f}"
        
        print(f"   {timestamp} | {signal} {quantity} shares @ {price} | Pred: {prediction}")

def generate_summary_report():
    """Generate a complete summary report"""
    print("📊 COMPLETE TRADING SUMMARY REPORT")
    print("="*80)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Calculate P&L
    total_pnl = calculate_pnl_from_trades()
    
    # Analyze accuracy
    analyze_model_accuracy()
    
    # Show detailed trades
    show_detailed_trades()
    
    print(f"\n" + "="*80)
    print("🎯 SUMMARY")
    print("="*80)
    print(f"💰 Total P&L: ${total_pnl:,.2f}")
    
    if total_pnl > 0:
        print("✅ PROFITABLE DAY!")
    elif total_pnl < 0:
        print("❌ LOSS DAY")
    else:
        print("➖ BREAK EVEN")
    
    print(f"\n💡 Next Steps:")
    print(f"   - Review your Alpaca dashboard")
    print(f"   - Check position status")
    print(f"   - Plan for tomorrow's trading")
    print(f"   - Consider adjusting strategy if needed")

if __name__ == "__main__":
    generate_summary_report() 