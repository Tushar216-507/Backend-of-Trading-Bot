from db import get_trades
from trader import api
import datetime

def simple_test():
    """Simple test of the results functionality"""
    print("🧪 Testing Trading Results Script")
    print("="*50)
    
    # Test database connection
    print("📊 Testing database connection...")
    try:
        trades = get_trades()
        print(f"✅ Found {len(trades)} trades in database")
        
        if trades:
            print("📋 Sample trade:")
            sample_trade = trades[0]
            for key, value in sample_trade.items():
                if key != '_id':  # Skip MongoDB ID
                    print(f"   {key}: {value}")
        else:
            print("📋 No trades found in database")
            
    except Exception as e:
        print(f"❌ Database error: {e}")
    
    # Test Alpaca connection
    print(f"\n💰 Testing Alpaca connection...")
    try:
        account = api.get_account()
        print(f"✅ Account Status: {account.status}")
        print(f"💰 Buying Power: ${float(account.buying_power):,.2f}")
        print(f"💵 Cash: ${float(account.cash):,.2f}")
        
        positions = api.list_positions()
        print(f"📈 Current Positions: {len(positions)}")
        
        if positions:
            for pos in positions:
                print(f"   {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")
        
    except Exception as e:
        print(f"❌ Alpaca error: {e}")
    
    print(f"\n✅ Test completed!")

if __name__ == "__main__":
    simple_test() 