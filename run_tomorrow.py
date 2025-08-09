import datetime
import time
from live_simulation import run_live_simulation, quick_test

def main():
    print("🚀 Tomorrow's Live Trading Simulation Setup")
    print("="*60)
    
    # Check if tomorrow is a weekday
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    is_weekday = tomorrow.weekday() < 5  # Monday = 0, Friday = 4
    
    print(f"📅 Tomorrow: {tomorrow.strftime('%A, %B %d, %Y')}")
    print(f"📊 Market Day: {'Yes' if is_weekday else 'No'}")
    
    if not is_weekday:
        print("❌ Tomorrow is not a trading day (weekend)")
        print("💡 Markets are closed on weekends")
        return
    
    print("\n✅ Tomorrow is a trading day!")
    print("📋 Live Trading Schedule:")
    print("   - Market Opens: 9:30 AM ET")
    print("   - Market Closes: 4:00 PM ET")
    print("   - Trading Interval: Every 5 minutes")
    print("   - Stock: AAPL")
    print("   - Risk per trade: 1% of buying power")
    
    print("\n🧪 Testing system readiness...")
    quick_test()
    
    print("\n" + "="*60)
    print("🎯 TO RUN TOMORROW'S LIVE SIMULATION:")
    print("="*60)
    print("1. Open terminal/command prompt")
    print("2. Navigate to your project folder:")
    print("   cd C:\\Users\\admin\\Desktop\\Xgboost_backen")
    print("3. Run the live simulation:")
    print("   py live_simulation.py")
    print("4. The system will:")
    print("   - Check if market is open")
    print("   - Make predictions every 5 minutes")
    print("   - Execute trades automatically")
    print("   - Save all trades to database")
    print("5. Press Ctrl+C to stop the simulation")
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("- This is PAPER TRADING (no real money)")
    print("- Monitor the system during market hours")
    print("- Check your Alpaca dashboard for orders")
    print("- All trades are logged in your database")
    
    print("\n💡 TIPS FOR TOMORROW:")
    print("- Start the script before 9:30 AM ET")
    print("- Keep the terminal window open")
    print("- Monitor the predictions and trades")
    print("- Check your account balance changes")
    
    print("\n🚀 Ready for tomorrow's live trading!")

if __name__ == "__main__":
    main() 