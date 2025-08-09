from db import get_trades
from trader import api
import datetime
import pytz

def _to_eastern(dt: datetime.datetime) -> str:
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        eastern = pytz.timezone('US/Eastern')
        return dt.astimezone(eastern).strftime('%Y-%m-%d %H:%M:%S %Z')
    except Exception:
        return str(dt)

def _get_order_info(order_id: str):
    try:
        order = api.get_order(order_id)
        filled = getattr(order, 'filled_at', None)
        filled_at = None
        if filled:
            # Alpaca may return ISO string
            try:
                filled_at = datetime.datetime.fromisoformat(str(filled).replace('Z', '+00:00'))
            except Exception:
                filled_at = None
        info = {
            'status': getattr(order, 'status', 'unknown'),
            'filled_at': filled_at,
            'filled_avg_price': float(getattr(order, 'filled_avg_price', 'nan')) if getattr(order, 'filled_avg_price', None) else None
        }
        return info
    except Exception:
        return {'status': 'unknown', 'filled_at': None, 'filled_avg_price': None}

def _position_status(symbol: str):
    try:
        positions = api.list_positions()
        for pos in positions:
            if pos.symbol == symbol:
                qty = float(getattr(pos, 'qty', 0))
                return ('OPEN' if qty != 0 else 'CLOSED', qty)
        return ('CLOSED', 0.0)
    except Exception:
        return ('UNKNOWN', None)

def _today_eastern_bounds():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    start = datetime.datetime.combine(now.date(), datetime.time(0, 0, tzinfo=eastern))
    return eastern, start

def _last_filled_order(symbol: str, side: str, after_dt: datetime.datetime | None = None):
    """Find the most recent filled order on the given side for the symbol today (or after after_dt)."""
    try:
        eastern, start = _today_eastern_bounds()
        after = after_dt.astimezone(eastern) if isinstance(after_dt, datetime.datetime) else start
        orders = api.list_orders(status='all', after=after.isoformat(), limit=200, direction='desc')
        for o in orders:
            try:
                if getattr(o, 'symbol', None) != symbol:
                    continue
                if getattr(o, 'side', '') != side.lower():
                    continue
                if getattr(o, 'status', '') != 'filled':
                    continue
                return o
            except Exception:
                continue
    except Exception:
        return None
    return None

def last_trade_pl():
    print("📊 LAST TRADE P/L")
    print("="*40)
    trades = get_trades()
    if not trades:
        print("❌ No trades found in database.")
        return
    last_trade = trades[-1]
    symbol = last_trade.get('stock', 'AAPL')
    side = last_trade.get('signal', 'BUY')
    qty = last_trade.get('quantity', 0)
    entry_price = last_trade.get('price', None)
    timestamp = last_trade.get('timestamp', None)
    order_id = last_trade.get('order_id', None)
    
    if entry_price is None:
        print("❌ Last trade does not have a price.")
        return
    
    # Load order execution details for accurate time/price
    order_info = _get_order_info(order_id) if order_id else {'status': 'unknown', 'filled_at': None, 'filled_avg_price': None}
    exec_time_str = _to_eastern(order_info['filled_at']) if order_info['filled_at'] else (timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, datetime.datetime) else str(timestamp))
    exec_price = float(entry_price)
    if order_info['filled_avg_price'] is not None:
        exec_price = float(order_info['filled_avg_price'])

    # Check current position status
    pos_status, pos_qty = _position_status(symbol)
    
    # Get current market price for unrealized P/L
    try:
        latest_trade = api.get_latest_trade(symbol)
        current_price = float(latest_trade.price)
    except Exception as e:
        current_price = None
    
    # Compute P/L depending on side
    if side.upper() == 'BUY':
        # If we don't have a reliable order fill, fallback to Alpaca's most recent filled BUY for this symbol (after our timestamp)
        if order_info['filled_at'] is None or order_info['filled_avg_price'] is None:
            after_dt = timestamp if isinstance(timestamp, datetime.datetime) else None
            o = _last_filled_order(symbol, 'buy', after_dt)
            if o is not None:
                try:
                    filled_at = getattr(o, 'filled_at', None)
                    filled_dt = datetime.datetime.fromisoformat(str(filled_at).replace('Z', '+00:00')) if filled_at else None
                except Exception:
                    filled_dt = None
                fap = getattr(o, 'filled_avg_price', None)
                if fap is not None:
                    exec_price = float(fap)
                if filled_dt is not None:
                    exec_time_str = _to_eastern(filled_dt)
        if current_price is None:
            print("❌ Could not fetch current price for P/L.")
            return
        pl = (current_price - exec_price) * int(qty)
        print(f"Symbol: {symbol}")
        print(f"Side: BUY (open) | Order Status: {order_info['status']} | Position: {pos_status}")
        print(f"Quantity: {qty}")
        print(f"Executed At: {exec_time_str}")
        print(f"Entry Price: ${exec_price:.2f}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Unrealized P/L: ${pl:.2f}")
        print("✅ PROFIT" if pl > 0 else ("❌ LOSS" if pl < 0 else "➖ BREAK EVEN"))
    else:  # SELL -> try to compute realized P/L vs previous BUY
        # Find prior BUY for same symbol from DB first; fallback to Alpaca orders
        prior_buy = None
        for t in reversed(trades[:-1]):
            if t.get('stock') == symbol and str(t.get('signal', '')).upper() == 'BUY':
                # ignore skipped pseudo orders
                if str(t.get('order_id', '')).startswith('skipped_'):
                    continue
                prior_buy = t
                break
        if not prior_buy:
            o = _last_filled_order(symbol, 'buy')
            if o is not None:
                try:
                    filled_at = getattr(o, 'filled_at', None)
                    buy_time = datetime.datetime.fromisoformat(str(filled_at).replace('Z', '+00:00')) if filled_at else None
                except Exception:
                    buy_time = None
                buy_price = float(getattr(o, 'filled_avg_price', exec_price)) if getattr(o, 'filled_avg_price', None) else exec_price
                prior_buy = {'timestamp': buy_time, 'price': buy_price, 'order_id': getattr(o, 'id', None), 'quantity': qty}
        sell_price = exec_price
        sell_time_str = exec_time_str
        if prior_buy:
            buy_order_info = _get_order_info(prior_buy.get('order_id')) if prior_buy.get('order_id') else {'filled_at': None, 'filled_avg_price': None, 'status': 'unknown'}
            buy_exec_price = float(prior_buy.get('price', 0.0))
            if buy_order_info['filled_avg_price'] is not None:
                buy_exec_price = float(buy_order_info['filled_avg_price'])
            buy_time = buy_order_info['filled_at'] or prior_buy.get('timestamp')
            buy_time_str = _to_eastern(buy_time) if isinstance(buy_time, datetime.datetime) else str(buy_time)
            realized_pl = (sell_price - buy_exec_price) * int(min(qty or 0, prior_buy.get('quantity', 0) or 0) or qty or 0)
            # Duration
            if isinstance(buy_time, datetime.datetime) and isinstance(order_info['filled_at'], datetime.datetime):
                duration = order_info['filled_at'] - buy_time
                duration_str = str(duration).split('.')[0]
            else:
                duration_str = "N/A"
            print(f"Symbol: {symbol}")
            print(f"Side: SELL (close) | Order Status: {order_info['status']} | Position: {pos_status}")
            print(f"Quantity: {qty}")
            print(f"Entry Time: {buy_time_str}")
            print(f"Exit Time:  {sell_time_str}")
            print(f"Entry Price: ${buy_exec_price:.2f}")
            print(f"Exit Price:  ${sell_price:.2f}")
            print(f"Hold Duration: {duration_str}")
            print(f"Realized P/L: ${realized_pl:.2f}")
            print("✅ PROFIT" if realized_pl > 0 else ("❌ LOSS" if realized_pl < 0 else "➖ BREAK EVEN"))
        else:
            # No prior buy found; report sell execution details only
            print(f"Symbol: {symbol}")
            print(f"Side: SELL (no matching prior BUY found) | Position: {pos_status}")
            print(f"Quantity: {qty}")
            print(f"Executed At: {sell_time_str}")
            print(f"Sell Price: ${sell_price:.2f}")
            if current_price is not None:
                print(f"Current Price: ${current_price:.2f}")
            print("ℹ️ Unable to compute realized P/L without matching entry trade")

if __name__ == "__main__":
    last_trade_pl()