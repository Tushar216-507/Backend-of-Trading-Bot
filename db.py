from pymongo import MongoClient
import os

# ✅ Replace with your actual credentials
USERNAME = "tusharkadam1248"
PASSWORD = "tushar2122"
CLUSTER  = "mycluster.hwdfiam"
DATABASE_NAME = "bot_backend"

# 🔐 Encode special characters in PASSWORD if needed
MONGO_URI = f"mongodb+srv://{USERNAME}:{PASSWORD}@{CLUSTER}.mongodb.net/{DATABASE_NAME}?retryWrites=true&w=majority&appName=MyCluster"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
trades_collection = db["trades"]

def save_trade(trade_data):
    trades_collection.insert_one(trade_data)
    print("✅ Trade saved to MongoDB")

def get_trades():
    """Get all trades from database"""
    try:
        trades = list(trades_collection.find())
        return trades
    except Exception as e:
        print(f"❌ Error getting trades: {e}")
        return []
