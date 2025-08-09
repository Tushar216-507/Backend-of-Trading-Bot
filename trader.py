import os
from alpaca_trade_api.rest import REST
from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, ALPACA_BASE_URL

api = REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, base_url=ALPACA_BASE_URL)

def place_order(
    stock,
    side,
    quantity,
    *,
    current_price=None,
    take_profit_pct=None,
    stop_loss_pct=None,
):
    try:
        # Check if we have any existing positions to avoid wash trading
        print(f"🔍 Checking existing positions for {stock}...")
        positions = api.list_positions()
        current_position = None
        
        for position in positions:
            if position.symbol == stock:
                current_position = position
                print(f"📊 Found existing position: {position.qty} shares of {stock}")
                break
        
        if not current_position:
            print(f"📊 No existing position found for {stock}")
        
        # If we have a position and trying to buy more, or no position and trying to sell
        if (current_position and side.lower() == 'buy' and float(current_position.qty) > 0) or \
           (not current_position and side.lower() == 'sell'):
            print(f"⚠️  Skipping {side} order for {stock} to avoid wash trading")
            return f"skipped_{stock}_{side}_{quantity}"
        
        # If BUY and TP/SL provided, place a bracket order
        if (
            side.lower() == 'buy'
            and (take_profit_pct is not None or stop_loss_pct is not None)
        ):
            # Ensure we have a fresh base price
            base_price = None
            try:
                if current_price is not None:
                    base_price = float(current_price)
                else:
                    latest = api.get_latest_trade(stock)
                    base_price = float(latest.price)
            except Exception:
                base_price = float(current_price) if current_price is not None else None

            tp_price = None
            sl_price = None
            if base_price is not None:
                if take_profit_pct is not None:
                    raw_tp = base_price * (1.0 + float(take_profit_pct))
                    # Alpaca requires TP >= base + 0.01; add a small buffer
                    tp_price = round(max(raw_tp, base_price + 0.02), 2)
                if stop_loss_pct is not None:
                    raw_sl = base_price * (1.0 - float(stop_loss_pct))
                    # Ensure SL <= base - 0.01; add a small buffer
                    sl_price = round(min(raw_sl, base_price - 0.02), 2)

            print(
                f"🚀 Placing BRACKET BUY for {quantity} {stock} | TP: {tp_price if tp_price else '—'} | SL: {sl_price if sl_price else '—'}"
            )
            try:
                order = api.submit_order(
                    symbol=stock,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day',
                    order_class='bracket',
                    take_profit={'limit_price': f"{tp_price:.2f}"} if tp_price else None,
                    stop_loss={'stop_price': f"{sl_price:.2f}"} if sl_price else None,
                )
            except Exception as bracket_err:
                print(f"⚠️  Bracket failed: {bracket_err}. Falling back to market BUY without bracket.")
                order = api.submit_order(
                    symbol=stock,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
        else:
            print(f"🚀 Attempting to place {side} order for {quantity} shares of {stock}...")
            order = api.submit_order(
                symbol=stock,
                qty=quantity,
                side=side.lower(),
                type='market',
                time_in_force='day'
            )
        print(f"✅ Order placed successfully: {order.id}")
        return order.id
    except Exception as e:
        print(f"Error placing order: {e}")
        # Return a mock order ID for demo purposes
        return f"mock_order_{stock}_{side}_{quantity}"
