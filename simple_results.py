from db import get_trades
from trader import api
import datetime

def show_trading_results():
    """Show simple trading results"""
    print("📊 TRADING RESULTS SUMMARY")
    print("="*60)
    print(f"📅 Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Get account status
    print("💰 ACCOUNT STATUS:")
    try:
        account = api.get_account()
        print(f"   Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
    except Exception as e:
        print(f"   ❌ Error getting account: {e}")
    
    # Get current positions
    print(f"\n📈 CURRENT POSITIONS:")
    try:
        positions = api.list_positions()
        if positions:
            total_unrealized_pl = 0
            for pos in positions:
                symbol = pos.symbol
                qty = float(pos.qty)
                avg_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                unrealized_pl = float(pos.unrealized_pl)
                total_unrealized_pl += unrealized_pl
                
                print(f"   {symbol}: {qty} shares @ ${avg_price:.2f}")
                print(f"     Current Price: ${current_price:.2f}")
                print(f"     Unrealized P&L: ${unrealized_pl:,.2f}")
            
            print(f"\n   Total Unrealized P&L: ${total_unrealized_pl:,.2f}")
        else:
            print("   No current positions")
    except Exception as e:
        print(f"   ❌ Error getting positions: {e}")
    
    # Get trades from database
    print(f"\n📋 TRADE HISTORY:")
    try:
        trades = get_trades()
        print(f"   Total trades in database: {len(trades)}")
        
        if trades:
            # Show recent trades
            print(f"\n   Recent Trades:")
            for i, trade in enumerate(trades[-5:]):  # Last 5 trades
                timestamp = trade.get('timestamp', 'Unknown')
                signal = trade.get('signal', 'Unknown')
                quantity = trade.get('quantity', 'Unknown')
                price = trade.get('price', 'Unknown')
                
                if isinstance(timestamp, datetime.datetime):
                    timestamp = timestamp.strftime('%H:%M:%S')
                
                if isinstance(price, (int, float)):
                    price = f"${price:.2f}"
                
                print(f"     {timestamp} | {signal} {quantity} shares @ {price}")
    except Exception as e:
        print(f"   ❌ Error getting trades: {e}")
    
    # Summary
    print(f"\n" + "="*60)
    print("🎯 SUMMARY:")
    print("="*60)
    
    try:
        # Calculate total P&L
        total_pnl = 0
        positions = api.list_positions()
        for pos in positions:
            total_pnl += float(pos.unrealized_pl)
        
        print(f"💰 Total P&L: ${total_pnl:,.2f}")
        
        if total_pnl > 0:
            print("✅ PROFITABLE!")
        elif total_pnl < 0:
            print("❌ LOSS")
        else:
            print("➖ BREAK EVEN")
            
    except Exception as e:
        print(f"❌ Error calculating P&L: {e}")
    
    print(f"\n💡 To check detailed results:")
    print(f"   - Visit your Alpaca dashboard")
    print(f"   - Check order history")
    print(f"   - Review position status")

if __name__ == "__main__":
    show_trading_results() 