import yfinance as yf

# ティッカーシンボルを指定
ticker = "AMZN"

# データを取得
data = yf.download(ticker, start="2024-11-20", end="2025-01-01")  # 必要な範囲を指定

# Adj Closeのデータを表示
print(data["Adj Close"])
