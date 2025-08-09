import yfinance as yf

symbol = 'ADBE'  # Replace with any US stock symbol like MSFT, TSLA, AMZN
data = yf.download(symbol, start="2015-01-01", end="2025-08-06", interval='1d')

data.reset_index(inplace=True)
data.rename(columns={
    'Date': 'timestamp',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}, inplace=True)

data['stock'] = symbol
data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'stock']]
data.to_csv("historical_data_ADBE.csv", index=False)

print(f"✅ Saved {symbol} historical data to historical_data_ADBE.csv")
