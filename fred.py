"""import os
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
from dotenv import load_dotenv

# .envファイルからAPIキーを読み込む
load_dotenv("fred_api.env")
api_key = os.getenv("API_KEY")
print(f"API Key: {api_key}")  # 確認用の出力

# APIキーを渡してFREDクライアントを作成
fred = Fred(api_key=api_key)

# データ取得: 開始時期と終了時期を指定
start_date = "2010-01-01"  # データの取得開始日
end_date = "2020-12-31"    # データの取得終了日

data = fred.get_series('^SKEW', observation_start=start_date, observation_end=end_date)
print(data)  # 取得したデータの先頭を表示

# データをプロットする
plt.figure(figsize=(10, 6))
data.plot(title="S&P 500", xlabel="Date", ylabel="Value")
plt.grid(True)
plt.show()

plt.savefig("output.png")  # 現在のプロットを "output.png" というファイルに保存

"""

import optuna
import pandas as pd
import torch
from src.model import iTransformer, EarlyStopping
from src.data_create import data_Normalization, create_multivariate_dataset
from src.train import train
import yfinance as yf   
from torch import nn
import numpy as np
import wandb
import matplotlib.pyplot as plt
import time
from statsmodels.tsa.seasonal import STL # STL分解
from fredapi import Fred
from dotenv import load_dotenv
import ta
import os

# .envファイルからAPIキーを読み込む
load_dotenv("fred_api.env")
api_key = os.getenv("API_KEY")
print(f"API Key: {api_key}")  # 確認用の出力
# APIキーを渡してFREDクライアントを作成
fred = Fred(api_key=api_key)

# Download stock data
stock_code = 'AAPL'
start_date = '2012-05-18'
end_date = '2023-06-01'

df_stock = yf.download(stock_code, start=start_date, end=end_date)
if len(df_stock) == 0:
    raise Exception("No data fetched for the given stock code and date range.")

data = df_stock['Adj Close']
print(data)

# Apply STL decomposition to AAPL stock data
stl = STL(data, period=252, robust=True)
stl_series = stl.fit()

# Extract trend, seasonal, and residual components
stock_trend = stl_series.trend
stock_seasonal = stl_series.seasonal
stock_resid = stl_series.resid


"""print(stock_trend.shape)
print(stock_seasonal.shape)
print(stock_resid.shape)"""

dataframes_trend = {stock_code: stock_trend}
dataframes_seasonal = {stock_code: stock_seasonal}
dataframes_resid = {stock_code: stock_resid}

"""print(dataframes_trend)
print(dataframes_seasonal)"""


fred_tickers = ['CPIAUCSL','DTWEXBGS', 'VIXCLS', 'DFII10', 'T10Y2Y'] # FRED tickers

for ticker in fred_tickers:
    print(f"Fetching data for {ticker}...")
    df = fred.get_series(ticker, observation_start=start_date, observation_end=end_date)
    # NaNを含む行を削除
    df = df.dropna()
    if len(df) == 0:
        raise Exception(f"No data fetched for {ticker} for the given date range.")
    dataframes_trend[ticker] = df
    """print(ticker)"""
    print(df)
    dataframes_seasonal[ticker] = df
    dataframes_resid[ticker] = df

print(dataframes_trend)

# 終値と出来高データ
close_data = df_stock['Adj Close']  # pandas.Series
volume_data = df_stock['Volume']   # pandas.Series
close_data = close_data.iloc[:, 0]

# 念のため、データが 1 次元であることを確認
if not isinstance(close_data, pd.Series):
    raise ValueError("close_data must be a pandas.Series")

# ボリンジャーバンド
bb_indicator = ta.volatility.BollingerBands(close=close_data, window=20, window_dev=2)
df_stock['BB_Upper'] = bb_indicator.bollinger_hband()
df_stock['BB_Lower'] = bb_indicator.bollinger_lband()
df_stock['BB_Middle'] = bb_indicator.bollinger_mavg()

# MACD
macd_indicator = ta.trend.MACD(close=close_data)
df_stock['MACD'] = macd_indicator.macd()
df_stock['MACD_Signal'] = macd_indicator.macd_signal()
df_stock['MACD_Diff'] = macd_indicator.macd_diff()

# RSI (14日)
rsi_indicator = ta.momentum.RSIIndicator(close=close_data, window=14)
df_stock['RSI'] = rsi_indicator.rsi()

# 移動平均 (50日と200日)
df_stock['SMA_50'] = close_data.rolling(window=50).mean()
df_stock['SMA_200'] = close_data.rolling(window=200).mean()

# 結果の確認
print(df_stock[['BB_Upper', 'BB_Lower', 'BB_Middle', 'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'SMA_50', 'SMA_200']].head(200))

