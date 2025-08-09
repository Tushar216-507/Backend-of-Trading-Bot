from trader import api

def cancel_pending_orders():
    """Cancel all pending orders to resolve wash trading issues"""
    try:
        print("🔍 Checking for pending orders...")
        orders = api.list_orders(status='new')
        
        if not orders:
            print("✅ No pending orders found")
            return
        
        print(f"📋 Found {len(orders)} pending orders:")
        for order in orders:
            print(f"   {order.symbol}: {order.qty} {order.side} (ID: {order.id})")
        
        # Cancel all pending orders
        for order in orders:
            try:
                api.cancel_order(order.id)
                print(f"✅ Cancelled order: {order.symbol} {order.qty} {order.side}")
            except Exception as e:
                print(f"❌ Failed to cancel order {order.id}: {e}")
        
        print("🎉 All pending orders cancelled successfully!")
        
    except Exception as e:
        print(f"❌ Error cancelling orders: {e}")

if __name__ == "__main__":
    cancel_pending_orders() 