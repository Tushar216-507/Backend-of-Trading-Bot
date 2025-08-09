from db import get_trades

trades = get_trades()
total_pnl = sum(trade.get('pnl', 0) for trade in trades)  # Using .get() with default value
wins = sum(1 for t in trades if t.get('pnl', 0) > 0)  # Added .get() with default value
losses = len(trades) - wins

print(f"Total Profit/Loss: ₹{total_pnl}")
print(f"Number of trades: {len(trades)}")
print(f"Wins: {wins}, Losses: {losses}")
