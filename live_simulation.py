import time
import schedule
import datetime
from predictor import get_prediction
from trader import place_order, api
from db import save_trade, get_trades
import pandas as pd
import pytz
import numpy as np

def check_market_hours():
    """Check if market is currently open"""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    
    # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
    is_market_hours = market_open <= now <= market_close
    
    return is_weekday and is_market_hours

def get_account_status():
    """Get current account status"""
    try:
        account = api.get_account()
        return {
            'status': account.status,
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value)
        }
    except Exception as e:
        print(f"❌ Error getting account status: {e}")
        return None

def get_current_positions():
    """Get current positions"""
    try:
        positions = api.list_positions()
        return positions
    except Exception as e:
        print(f"❌ Error getting positions: {e}")
        return []

def to_python_type(val):
    if isinstance(val, (np.generic,)):
        return val.item()
    return val

def execute_live_trade():
    """Execute a live trade based on model prediction"""
    print(f"\n{'='*60}")
    print(f"🕐 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check if market is open
    if not check_market_hours():
        print("❌ Market is closed. Waiting for next check...")
        return
    
    print("✅ Market is open! Starting live trading simulation...")

    # Force-close any open positions each cycle to guarantee realized P/L every 5 minutes
    try:
        force_cycle_exit()
        # Also keep time-based exit as a backstop if anything remains
        enforce_time_based_exits(max_hold_minutes=5)
        poll_time_exits()
        poll_bracket_exits()
    except Exception:
        pass
    
    # Get account status
    account_status = get_account_status()
    if account_status:
        print(f"💰 Account Status:")
        print(f"   Status: {account_status['status']}")
        print(f"   Buying Power: ${account_status['buying_power']:,.2f}")
        print(f"   Cash: ${account_status['cash']:,.2f}")
        print(f"   Portfolio Value: ${account_status['portfolio_value']:,.2f}")
    
    # Get current positions
    positions = get_current_positions()
    if positions:
        print(f"📊 Current Positions:")
        for pos in positions:
            print(f"   {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")
    else:
        print("📊 No current positions")
    
    try:
        
        # Get model prediction
        print("\n🤖 Getting model prediction...")
        features, prediction = get_prediction()
        if prediction == -1:
            print("\n🛑 HOLD: Skipping trade due to low confidence or risk filters")
            return
        signal = "BUY" if prediction == 1 else "SELL"
        current_price = features['close']
        
        print(f"📈 Market Analysis:")
        print(f"   Stock: {features['stock']}")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Model Prediction: {prediction} ({signal})")
        if 'model_confidence' in features:
            print(f"   Model Confidence: {features['model_confidence']:.3f}")
        print(f"   RSI: {features['rsi']:.2f}")
        print(f"   MACD: {features['macd']:.4f}")
        print(f"   EMA10 vs EMA20: {features['ema10']:.2f} vs {features['ema20']:.2f}")
        
        # Calculate position size (1% of buying power per trade)
        if account_status and account_status['buying_power'] > 0:
            max_trade_value = account_status['buying_power'] * 0.01  # 1% risk per trade
            quantity = int(max_trade_value / current_price)
            quantity = max(1, min(quantity, 100))  # Between 1 and 100 shares
        else:
            quantity = 10  # Default quantity
        
        print(f"\n💼 Trade Decision:")
        print(f"   Signal: {signal}")
        print(f"   Quantity: {quantity} shares")
        print(f"   Trade Value: ${quantity * current_price:.2f}")
        
        # Basic gating: skip low-confidence BUY conditions handled upstream
        print(f"\n🚀 Executing {signal} order...")
        # Optional bracket exits: 0.4% TP, 0.3% SL (tune as desired)
        tp_pct = 0.004 if signal == 'BUY' else None
        sl_pct = 0.003 if signal == 'BUY' else None
        order_id = place_order(
            features['stock'],
            signal,
            quantity,
            current_price=current_price,
            take_profit_pct=tp_pct,
            stop_loss_pct=sl_pct,
        )
        
        # Record the trade
        trade_data = {
            "timestamp": datetime.datetime.now(),
            "stock": str(to_python_type(features['stock'])),
            "signal": str(to_python_type(signal)),
            "quantity": int(to_python_type(quantity)),
            "price": float(to_python_type(current_price)),
            "order_id": str(to_python_type(order_id)),
            "prediction": int(to_python_type(prediction)),
            "trade_type": "LIVE_SIMULATION"
        }
        # Add all model features for retraining
        model_features = [
            'return', 'volatility', 'ohlc_delta', 'candle_body',
            'ema10', 'ema20', 'rsi', 'macd',
            'bb_bbm', 'bb_bbh', 'bb_bbl',
            'volume_surge', 'vwap_deviation',
            'open_gap_pct', 'price_change', 'price_change_pct',
            'price_vs_sma50', 'price_vs_sma200',
            'hour', 'day_of_week'
        ]
        for feat in model_features:
            trade_data[feat] = float(to_python_type(features[feat]))
        
        # Save to database
        save_trade(trade_data)
        print(f"✅ Trade recorded in database")
        
        print(f"\n📊 Trade Summary:")
        print(f"   {signal} {quantity} shares of {features['stock']} at ${current_price:.2f}")
        print(f"   Order ID: {order_id}")
        print(f"   Trade Value: ${quantity * current_price:.2f}")
        
    except Exception as e:
        print(f"❌ Error in live trading: {e}")
        print("🔧 Continuing with next iteration...")

ANNOUNCED_LEG_IDS = set()
TIME_EXIT_PENDING = {}

def poll_bracket_exits():
    """Check for filled take-profit/stop-loss legs and print realized P/L once."""
    try:
        trades = get_trades()
        if not trades:
            return
        eastern = pytz.timezone('US/Eastern')
        today = datetime.datetime.now(eastern).date()
        for t in trades:
            if t.get('trade_type') != 'LIVE_SIMULATION':
                continue
            if str(t.get('signal', '')).upper() != 'BUY':
                continue
            order_id = t.get('order_id')
            if not order_id:
                continue
            try:
                order = api.get_order(order_id)
            except Exception:
                continue
            legs = getattr(order, 'legs', None) or []
            for leg in legs:
                try:
                    if getattr(leg, 'status', '') != 'filled' or getattr(leg, 'side', '') != 'sell':
                        continue
                    leg_id = getattr(leg, 'id', None)
                    if not leg_id or leg_id in ANNOUNCED_LEG_IDS:
                        continue
                    # Only announce same-day exits to reduce noise
                    filled_at = getattr(leg, 'filled_at', None)
                    exit_time = None
                    if filled_at:
                        try:
                            exit_time = datetime.datetime.fromisoformat(str(filled_at).replace('Z', '+00:00'))
                        except Exception:
                            exit_time = None
                    if exit_time and exit_time.astimezone(eastern).date() != today:
                        continue
                    # Compute realized P/L
                    try:
                        entry_px = float(getattr(order, 'filled_avg_price', t.get('price', 0.0)))
                    except Exception:
                        entry_px = float(t.get('price', 0.0) or 0.0)
                    try:
                        exit_px = float(getattr(leg, 'filled_avg_price', 0.0))
                    except Exception:
                        continue
                    try:
                        qty = int(getattr(leg, 'qty', t.get('quantity', 0)) or 0)
                    except Exception:
                        qty = int(t.get('quantity', 0) or 0)
                    if qty <= 0:
                        continue
                    realized = (exit_px - entry_px) * qty
                    ts_str = exit_time.astimezone(eastern).strftime('%Y-%m-%d %H:%M:%S %Z') if exit_time else 'N/A'
                    print("\n🎯 BRACKET EXIT FILLED")
                    print(f"   Symbol: {t.get('stock', 'AAPL')} | Qty: {qty}")
                    print(f"   Entry: ${entry_px:.2f}  →  Exit: ${exit_px:.2f}")
                    print(f"   Time:  {ts_str}")
                    print(f"   Realized P/L: ${realized:.2f}")
                    print("✅ PROFIT" if realized > 0 else ("❌ LOSS" if realized < 0 else "➖ BREAK EVEN"))
                    ANNOUNCED_LEG_IDS.add(leg_id)
                except Exception:
                    continue
    except Exception:
        # Silent; this is a best-effort notifier
        pass

def poll_time_exits():
    """Announce realized P/L for time-based market SELL orders once they are filled."""
    try:
        eastern = pytz.timezone('US/Eastern')
        pending_ids = list(TIME_EXIT_PENDING.keys())
        for oid in pending_ids:
            try:
                order = api.get_order(oid)
                if getattr(order, 'status', '') != 'filled':
                    continue
                ctx = TIME_EXIT_PENDING.get(oid)
                if not ctx:
                    continue
                exit_px = float(getattr(order, 'filled_avg_price', 0.0) or 0.0)
                entry_px = float(ctx['entry_price'])
                qty = int(ctx['qty'])
                symbol = ctx['symbol']
                realized = (exit_px - entry_px) * qty
                filled_at = getattr(order, 'filled_at', None)
                exit_time = None
                if filled_at:
                    try:
                        exit_time = datetime.datetime.fromisoformat(str(filled_at).replace('Z', '+00:00'))
                    except Exception:
                        exit_time = None
                et_str = exit_time.astimezone(eastern).strftime('%Y-%m-%d %H:%M:%S %Z') if isinstance(exit_time, datetime.datetime) else 'N/A'
                print("\n🎯 TIME-BASED EXIT FILLED")
                print(f"   Symbol: {symbol} | Qty: {qty}")
                print(f"   Entry: ${entry_px:.2f}  →  Exit: ${exit_px:.2f}")
                print(f"   Time:  {et_str}")
                print(f"   Realized P/L: ${realized:.2f}")
                print("✅ PROFIT" if realized > 0 else ("❌ LOSS" if realized < 0 else "➖ BREAK EVEN"))
                TIME_EXIT_PENDING.pop(oid, None)
            except Exception:
                continue
    except Exception:
        pass

def force_cycle_exit():
    """Force-close all open positions at market each cycle before new trades."""
    try:
        positions = api.list_positions()
    except Exception:
        positions = []
    if not positions:
        return
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    for pos in positions:
        try:
            symbol = pos.symbol
            qty = int(float(pos.qty))
            if qty <= 0:
                continue
            # Cancel any child legs just in case
            try:
                # Find latest parent order for this symbol today (BUY)
                after = datetime.datetime.combine(now.date(), datetime.time(0, 0, tzinfo=eastern))
                orders = api.list_orders(status='open', after=after.isoformat(), limit=200, direction='desc')
                for o in orders:
                    if getattr(o, 'symbol', None) != symbol:
                        continue
                    for leg in getattr(o, 'legs', []) or []:
                        st = getattr(leg, 'status', '')
                        if st in ('new', 'accepted'):
                            try:
                                api.cancel_order(getattr(leg, 'id', None))
                            except Exception:
                                pass
            except Exception:
                pass
            # Price snapshot for estimate
            try:
                latest_trade = api.get_latest_trade(symbol)
                current_price = float(latest_trade.price)
            except Exception:
                current_price = float(getattr(pos, 'avg_entry_price', 0.0) or 0.0)
            entry_price = float(getattr(pos, 'avg_entry_price', 0.0) or 0.0)
            realized_est = (current_price - entry_price) * qty
            sell_order = api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
            print("\n⛔ FORCED CYCLE EXIT")
            print(f"   Symbol: {symbol} | Qty: {qty}")
            print(f"   Entry: ${entry_price:.2f}  →  Exit (mkt est): ${current_price:.2f}")
            print(f"   Realized P/L (est): ${realized_est:.2f}")
            print(f"   Order ID: {sell_order.id}")
            TIME_EXIT_PENDING[str(sell_order.id)] = {
                'symbol': symbol,
                'qty': qty,
                'entry_price': entry_price,
                'entry_time': now,
            }
        except Exception:
            continue

def _get_last_live_buy(symbol: str):
    """Return the most recent LIVE_SIMULATION BUY trade for symbol from DB."""
    try:
        trades = get_trades()
        if not trades:
            return None
        df = pd.DataFrame(trades)
        if 'trade_type' not in df.columns or 'signal' not in df.columns:
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        sub = df[(df['trade_type'] == 'LIVE_SIMULATION') & (df['signal'] == 'BUY') & (df['stock'] == symbol)]
        # Ignore simulated/skipped orders
        if 'order_id' in sub.columns:
            sub = sub[~sub['order_id'].astype(str).str.startswith('skipped_')]
        if sub.empty:
            return None
        sub = sub.sort_values('timestamp')
        return sub.iloc[-1].to_dict()
    except Exception:
        return None

def _get_last_filled_buy_from_alpaca(symbol: str):
    """Fallback: find last filled BUY order for symbol from Alpaca order history (today)."""
    try:
        eastern = pytz.timezone('US/Eastern')
        today = datetime.datetime.now(eastern).date()
        after = datetime.datetime.combine(today, datetime.time(0, 0), tzinfo=eastern)
        orders = api.list_orders(status='all', after=after.isoformat(), limit=200, direction='desc')
        for o in orders:
            try:
                if getattr(o, 'symbol', None) != symbol:
                    continue
                if getattr(o, 'side', '') != 'buy':
                    continue
                if getattr(o, 'status', '') != 'filled':
                    continue
                filled_at = getattr(o, 'filled_at', None)
                fap = getattr(o, 'filled_avg_price', None)
                filled_dt = None
                if filled_at:
                    try:
                        filled_dt = datetime.datetime.fromisoformat(str(filled_at).replace('Z', '+00:00'))
                    except Exception:
                        filled_dt = None
                entry_price = float(fap) if fap is not None else None
                if filled_dt and entry_price is not None:
                    return {'timestamp': filled_dt, 'order_id': getattr(o, 'id', None), 'price': entry_price}
            except Exception:
                continue
    except Exception:
        pass
    return None

def enforce_time_based_exits(max_hold_minutes: int = 5):
    """If a position has been open for >= max_hold_minutes, close it at market and print realized P/L estimate."""
    try:
        positions = api.list_positions()
    except Exception:
        positions = []
    if not positions:
        return
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    for pos in positions:
        try:
            symbol = pos.symbol
            qty = int(float(pos.qty))
            if qty <= 0:
                continue
            # Prefer DB record; fallback to Alpaca order history
            last_buy = _get_last_live_buy(symbol)
            if not last_buy:
                last_buy = _get_last_filled_buy_from_alpaca(symbol)
            if not last_buy:
                # Fallback: unknown entry (position might be opened outside this bot). Close immediately using avg entry.
                try:
                    entry_price = float(getattr(pos, 'avg_entry_price', 0.0) or 0.0)
                except Exception:
                    entry_price = 0.0
                # Get current price
                try:
                    latest_trade = api.get_latest_trade(symbol)
                    current_price = float(latest_trade.price)
                except Exception:
                    current_price = entry_price
                realized_est = (current_price - entry_price) * qty
                sell_order = api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
                print("\n⏱️ TIME-BASED EXIT (FALLBACK)")
                print(f"   Symbol: {symbol} | Qty: {qty}")
                print(f"   Entry (avg): ${entry_price:.2f}")
                print(f"   Exit (mkt):  ${current_price:.2f} at {now.strftime('%H:%M:%S %Z')}")
                print(f"   Realized P/L (est): ${realized_est:.2f}")
                print(f"   Order ID: {sell_order.id}")
                TIME_EXIT_PENDING[str(sell_order.id)] = {
                    'symbol': symbol,
                    'qty': qty,
                    'entry_price': entry_price,
                    'entry_time': now,
                }
                continue
            ts = last_buy.get('timestamp')
            order_id = last_buy.get('order_id')
            # Resolve entry time and price
            entry_time = None
            entry_price = float(last_buy.get('price', 0.0) or 0.0)
            if isinstance(ts, str):
                try:
                    entry_time = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except Exception:
                    entry_time = None
            elif isinstance(ts, datetime.datetime):
                entry_time = ts
            # Prefer Alpaca filled_avg_price if available
            try:
                if order_id:
                    order = api.get_order(order_id)
                    fap = getattr(order, 'filled_avg_price', None)
                    if fap is not None:
                        entry_price = float(fap)
            except Exception:
                pass
            if entry_time is None:
                continue
            # Normalize to Eastern (assume naive timestamps are already in Eastern/local trading time)
            if entry_time.tzinfo is None:
                try:
                    entry_time_et = eastern.localize(entry_time)
                except Exception:
                    entry_time_et = entry_time
            else:
                entry_time_et = entry_time.astimezone(eastern)
            age_min = (now - entry_time_et).total_seconds() / 60.0
            # Add a small buffer (5 seconds) to avoid boundary issues
            if age_min < (max_hold_minutes - 1/12):
                continue
            # Cancel any remaining bracket legs to avoid conflicts
            try:
                if order_id:
                    parent = api.get_order(order_id)
                    for leg in getattr(parent, 'legs', []) or []:
                        st = getattr(leg, 'status', '')
                        if st in ('new', 'accepted'):
                            try:
                                api.cancel_order(getattr(leg, 'id', None))
                            except Exception:
                                pass
            except Exception:
                pass
            # Get current price for P/L estimate
            try:
                latest_trade = api.get_latest_trade(symbol)
                current_price = float(latest_trade.price)
            except Exception:
                current_price = entry_price
            realized_est = (current_price - entry_price) * qty
            # Submit market SELL to realize P/L
            try:
                sell_order = api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
            except Exception as e:
                print(f"❌ Time exit SELL failed: {e}. Retrying once...")
                time.sleep(1)
                sell_order = api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
            print("\n⏱️ TIME-BASED EXIT")
            print(f"   Symbol: {symbol} | Qty: {qty}")
            print(f"   Entry: ${entry_price:.2f} at {entry_time_et.strftime('%H:%M:%S %Z')}")
            print(f"   Exit (mkt): ${current_price:.2f} at {now.strftime('%H:%M:%S %Z')}")
            print(f"   Hold: {int(age_min)} min | Realized P/L (est): ${realized_est:.2f}")
            print(f"   Order ID: {sell_order.id}")
            # Track for actual fill announcement
            TIME_EXIT_PENDING[str(sell_order.id)] = {
                'symbol': symbol,
                'qty': qty,
                'entry_price': entry_price,
                'entry_time': entry_time_et,
            }
        except Exception:
            continue

def run_live_simulation():
    """Run the live trading simulation"""
    print("🚀 Starting Live Trading Simulation")
    print("="*60)
    print("📋 Configuration:")
    print("   - Trading Interval: Every 5 minutes")
    print("   - Stock: AAPL")
    print("   - Risk per trade: 1% of buying power")
    print("   - Market hours: 9:30 AM - 4:00 PM ET")
    print("   - Days: Monday-Friday")
    print("="*60)
    
    # Schedule trades every 5 minutes during market hours
    schedule.every(5).minutes.do(execute_live_trade)
    
    # Run initial trade
    execute_live_trade()
    
    print("\n⏰ Live simulation running... Press Ctrl+C to stop")
    print("📊 Trades will execute every 5 minutes during market hours")
    
    try:
        while True:
            schedule.run_pending()
            # Poll for bracket exits frequently
            poll_bracket_exits()
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Live simulation stopped by user")
        print("📊 Final summary will be displayed...")

def quick_test():
    """Quick test to verify everything works"""
    print("🧪 Quick Test - Verifying Live Trading Setup")
    print("="*50)
    
    # Test market hours check
    is_open = check_market_hours()
    print(f"Market Open: {is_open}")
    
    # Test account status
    account = get_account_status()
    if account:
        print(f"Account Status: {account['status']}")
        print(f"Buying Power: ${account['buying_power']:,.2f}")
    
    # Test model prediction
    try:
        features, prediction = get_prediction()
        print(f"Model Prediction: {prediction} ({'BUY' if prediction == 1 else 'SELL'})")
        print(f"Current Price: ${features['close']:.2f}")
        print("✅ All systems ready for live trading!")
    except Exception as e:
        print(f"❌ Error in prediction: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        run_live_simulation() 