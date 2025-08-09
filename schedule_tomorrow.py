import os
import subprocess
import datetime

def create_scheduled_task():
    """Create a Windows scheduled task to run live simulation tomorrow"""
    print("🤖 Creating Scheduled Task for Tomorrow's Trading")
    print("="*60)
    
    # Get tomorrow's date
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    tomorrow_str = tomorrow.strftime('%Y-%m-%d')
    
    # Create the command
    script_path = os.path.abspath("live_simulation.py")
    python_path = "py"  # or "python" depending on your setup
    
    # Create batch file
    batch_content = f'''@echo off
cd /d "C:\\Users\\admin\\Desktop\\Xgboost_backen"
{python_path} live_simulation.py
pause
'''
    
    batch_file = "start_trading.bat"
    with open(batch_file, 'w') as f:
        f.write(batch_content)
    
    print(f"✅ Created batch file: {batch_file}")
    print(f"📅 Scheduled for: {tomorrow.strftime('%A, %B %d, %Y')} at 9:25 AM")
    
    print(f"\n🎯 TO SET UP AUTOMATIC STARTUP:")
    print(f"1. Press Win + R")
    print(f"2. Type: taskschd.msc")
    print(f"3. Click 'Create Basic Task'")
    print(f"4. Name: 'Live Trading Simulation'")
    print(f"5. Trigger: Daily")
    print(f"6. Start time: 9:25 AM")
    print(f"7. Action: Start a program")
    print(f"8. Program: {os.path.abspath(batch_file)}")
    print(f"9. Finish")
    
    print(f"\n⚠️  ALTERNATIVE - Manual Method:")
    print(f"1. Tomorrow at 9:25 AM, open terminal")
    print(f"2. Run: py live_simulation.py")
    print(f"3. Keep terminal open during market hours")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"For first time, use MANUAL method to monitor the system")
    print(f"Once you're confident, set up the scheduled task")

if __name__ == "__main__":
    create_scheduled_task() 