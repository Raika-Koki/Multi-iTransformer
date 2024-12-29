import os
import ta
import time
import json
import torch
import pandas as pd
import yfinance as yf
import torch.nn as nn
import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from fredapi import Fred
from datetime import datetime
from pandas.tseries.offsets import BDay
from dotenv import load_dotenv
from statsmodels.tsa.seasonal import MSTL
from src.model import iTransformer, EarlyStopping
from src.data_create import data_Normalization, create_multivariate_dataset
from src.train import train

# デバイスの設定
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 必要に応じて変更してください
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# モデルの定義
def create_model(params, num_variates, predict_period_num, depth, dim):
    model = iTransformer(
        num_variates=num_variates,
        lookback_len=params['observation_period_num'],
        depth=depth,
        dim=dim,
        pred_length=predict_period_num
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    return model, optimizer, scheduler

def decompose_stock_data(data, periods, iterate, stl_kwargs):
    """
    MSTLを使用して時系列データを分解する関数

    Parameters:
    - data: pd.Series, 時系列データ
    - periods: list, 季節性周期のリスト
    - iterate: int, MSTLの反復回数
    - stl_kwargs: dict, STLのオプション設定

    Returns:
    - result: MSTLオブジェクト, 分解結果
    """
    mstl = MSTL(data, periods=periods, iterate=iterate, stl_kwargs=stl_kwargs)
    result = mstl.fit()
    return result

def preprocess_open_data(data_open, reference_index):
    data_open.index = data_open.index - BDay(1)    
    data_open = data_open.dropna().reindex(reference_index).interpolate(method='linear').ffill()
    data_open.index = data_open.index.tz_localize(None)
    return data_open

# 指標ごとに関心度を計算する関数
def calculate_interest_levels(release_dates, date_range, alpha, beta_param, weight):
    interest_levels = pd.DataFrame({"Date": date_range, "Interest_Level": 0.0})
    beta_values_max = 0

    for release_date in release_dates:
        # 発表日の前後5日間の範囲
        day_offsets = (interest_levels["Date"] - release_date).dt.days
        within_range = day_offsets.between(-7, 2)
        
        # ベータ分布のスケールとシフトを適用
        x_values = (day_offsets[within_range] + 8) / 10  # 発表日 (-5, +5) を (0, 1) に変換

        if len(x_values) > 0:
            # ベータ分布に基づいて関心度を計算
            beta_values = beta.pdf(x_values, alpha, beta_param)
            if beta_values_max == 0:
                beta_values_max = beta_values.max()

            beta_values *= weight / beta_values_max  # 重みに基づき正規化

            # 発表日周辺の日付に関心度を割り当て
            interest_levels.loc[within_range, "Interest_Level"] += beta_values

    return interest_levels


# .envファイルからAPIキーを読み込む
load_dotenv("api_key.env")
fred_api_key = os.getenv("FRED_API")
fred = Fred(api_key=fred_api_key)

# JSONファイルからハイパーパラメータと日付を読み込む
with open("best_hyperparameters_AMZN_1.json", "r") as f:
    best_hyperparams = json.load(f)

# 学習期間の設定
dates_info = best_hyperparams["2012-05-18_to_2023-06-01"]["dates"]
start_date = dates_info["start_date"]
end_date = dates_info["end_date"]

# 経済指標ごとに関心度を計算
alpha = 8
beta_param = 3
weights = {
    'CPI_Dates': 0.9,
    'PCE_Dates': 0.8,
    'PPI_Dates': 0.7,
    'Unemployment_Rate_Dates': 0.8,
    'Nonfarm_Payrolls_Release_Dates': 0.9,
    'Monetary_Base_Data_Dates': 0.6
}

# データの取得
stock_code = 'AMZN'
df_stock = yf.download(stock_code, start=start_date, end=end_date)
if df_stock.empty:
    print("株価データが取得できませんでした。")
    exit()

depth = 32
dim = 64


# 学習時間の計測
start_time = time.time()

# 調整済み終値と始値の取得
data_Adj_close = df_stock['Adj Close'].tz_localize(None)
next_start_date = pd.to_datetime(start_date)
next_end_date = pd.to_datetime(end_date)
df_stock_dt = yf.download(stock_code, start=next_start_date + BDay(1), end = next_end_date + BDay(1))
if len(df_stock_dt) == 0:
    print(f"エンド日 {end_date} にデータが取得できません。処理を終了します。")
    exit()

data_Open = df_stock_dt['Open'].tz_localize(None)
data_Open = preprocess_open_data(data_Open, data_Adj_close.index)

periods = [252, 504, 756, 1260]
iterate = 3
stl_kwargs = {"seasonal_deg": 0, "inner_iter": 3, "outer_iter": 0}

mstl_series = decompose_stock_data(data_Adj_close, periods, iterate, stl_kwargs)
mstl_series_open = decompose_stock_data(data_Open, periods, iterate, stl_kwargs)

# トレンド、季節性、残差の成分を抽出
stock_trend = mstl_series.trend.tz_localize(None)
stock_seasonal_0, stock_seasonal_1, stock_seasonal_2, stock_seasonal_3 = [
    mstl_series.seasonal.iloc[:, i].tz_localize(None) for i in range(4)
]
stock_resid = mstl_series.resid.tz_localize(None)

stock_trend_open = mstl_series_open.trend.tz_localize(None)
stock_seasonal_0_open, stock_seasonal_1_open, stock_seasonal_2_open, stock_seasonal_3_open = [
    mstl_series_open.seasonal.iloc[:, i].tz_localize(None) for i in range(4)
]
stock_resid_open = mstl_series_open.resid.tz_localize(None)


# 現在の end_date に基づいたデータフレームを更新
dataframes_trend = {"Adj_close": stock_trend, "open": stock_trend_open}
dataframes_seasonal_0 = {"Adj_close": stock_seasonal_0, "open": stock_seasonal_0_open}
dataframes_seasonal_1 = {"Adj_close": stock_seasonal_1, "open": stock_seasonal_1_open}
dataframes_seasonal_2 = {"Adj_close": stock_seasonal_2, "open": stock_seasonal_2_open}
dataframes_seasonal_3 = {"Adj_close": stock_seasonal_3, "open": stock_seasonal_3_open}
dataframes_resid = {"Adj_close": stock_resid, "open": stock_resid_open}

# 指標データの取得と前処理
indicator_release_days = ['CPI_Dates', 'PCE_Dates', 'PPI_Dates', 'Unemployment_Rate_Dates',
                          'Nonfarm_Payrolls_Release_Dates', 'Monetary_Base_Data_Dates']
fred_tickers = ['DTWEXBGS', 'VIXCLS', 'DFII10', 'T10Y2Y', 'DCOILWTICO']
tech_indicator = ['Volume', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'MACD', 'MACD_Signal',
                  'MACD_Diff', 'RSI', 'SMA_50', 'SMA_200', 'SMA_200-50']

# 経済指標のデータを取得
with open('Release_Dates.json', 'r') as file:
    release_dates_data = json.load(file)

date_range = pd.date_range(start=start_date, end=end_date)

# 指標ごとに関心度を計算して出力
for indicator in indicator_release_days:
    release_dates = []
    found_post_end_date = False
    pre_start_date_release_date = None
    for year in release_dates_data[indicator]:
        for month in release_dates_data[indicator][year]:
            release_date = release_dates_data[indicator][year][month]
            if start_date <= release_date <= end_date:
                release_dates.append(release_date)
            elif release_date > end_date and not found_post_end_date:
                release_dates.append(release_date)
                found_post_end_date = True
            elif release_date < start_date:
                if pre_start_date_release_date is None or release_date > pre_start_date_release_date:
                    pre_start_date_release_date = release_date

    if pre_start_date_release_date:
        release_dates.append(pre_start_date_release_date)
        release_dates = [ (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d') for date in release_dates ]
    release_dates = pd.to_datetime(release_dates)
    weight = weights[indicator]
    interest_levels = calculate_interest_levels(release_dates, date_range, alpha, beta_param, weight)

    # 'Date'列を日付型に変換してインデックスに設定
    interest_levels['Date'] = pd.to_datetime(interest_levels['Date'])
    interest_levels.set_index('Date', inplace=True)
    
    # interest_levels のデータをフィルタリングして df_stock の日付に一致するものだけ残す
    interest_levels = interest_levels.loc[interest_levels.index.intersection(df_stock.index.tz_localize(None))]
    interest_levels = interest_levels.iloc[:, 0]  # 1列目のみ残す

    # 結果を保存
    dataframes_trend[indicator] = interest_levels
    dataframes_seasonal_0[indicator] = interest_levels
    dataframes_seasonal_1[indicator] = interest_levels
    dataframes_seasonal_2[indicator] = interest_levels
    dataframes_seasonal_3[indicator] = interest_levels
    dataframes_resid[indicator] = interest_levels

    # FRED APIからデータを取得するためのマッピング
    indicator_to_fred_mapping = {
        'CPI_Dates': ['CPIAUCSL', 'CPILFESL'],
        'PCE_Dates': ['PCEPI', 'PCEPILFE'],
        'PPI_Dates': ['PPIACO', 'WPSFD4131'],
        'Unemployment_Rate_Dates': ['UNRATE'],
        'Nonfarm_Payrolls_Release_Dates': ['PAYEMS'],
        'Monetary_Base_Data_Dates': ['BOGMBASE']#,
        #'Federal_Funds_Rate_Release_Dates': ['FEDFUNDS']
    }

    # indicator_release_daysの各指標についてFREDからデータを取得
    fred_keys = indicator_to_fred_mapping.get(indicator, [])
    for fred_key in fred_keys:
        # start_dateの前の月までデータを取得
        start_date_extended = (pd.to_datetime(start_date) - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        indicator_data = fred.get_series(fred_key, observation_start=start_date_extended, observation_end=end_date)
        if len(indicator_data) == 0:
            raise Exception(f"No data fetched for {fred_key} for the given date range.")
        
        # indicator_dataの日付をRelease_Dates.jsonの日付に変更
        new_index = []
        for date in indicator_data.index:
            year = date.strftime('%Y')
            month = date.strftime('%B')
            if year in release_dates_data[indicator]:
                if month in release_dates_data[indicator][year]:
                    release_date_str = release_dates_data[indicator][year][month]
                    release_date = pd.to_datetime(release_date_str)
                    new_index.append(release_date)
                else:
                    new_index.append(date)
            else:
                new_index.append(date)
        indicator_data.index = pd.to_datetime(new_index)
        indicator_data.index -= pd.Timedelta(days=1)

        # print(f'{fred_key}', indicator_data)
        # タイムゾーン解除
        indicator_data.index = indicator_data.index.tz_localize(None)
        # Reindex indicator_data to match df_stock's dates and forward fill values
        indicator_data = indicator_data.reindex(df_stock.index.tz_localize(None), method='ffill')
        
        #print(f'{fred_key}', indicator_data)

        # 結果を保存
        dataframes_trend[f"{indicator}_{fred_key}"] = indicator_data
        dataframes_seasonal_0[f"{indicator}_{fred_key}"] = indicator_data
        dataframes_seasonal_1[f"{indicator}_{fred_key}"] = indicator_data
        dataframes_seasonal_2[f"{indicator}_{fred_key}"] = indicator_data
        dataframes_seasonal_3[f"{indicator}_{fred_key}"] = indicator_data
        dataframes_resid[f"{indicator}_{fred_key}"] = indicator_data

#print(dataframes_trend)


for ticker in fred_tickers:
    # FREDデータの取得
    df = fred.get_series(ticker, observation_start=start_date, observation_end=end_date)
    if len(df) == 0:
        raise Exception(f"No data fetched for {ticker} for the given date range.")
    
    
    # タイムゾーン解除
    df.index = df.index.tz_localize(None)
    # 欠損データ削除（取得後の前処理）
    df = df.dropna()
    # 全営業日範囲を作成
    full_date_range = pd.date_range(start=df_stock.index.min(), end=df_stock.index.max(), freq='B')
    # リインデックス時にタイムゾーン解除を適用
    df = df.reindex(full_date_range.tz_localize(None))
    # 補完処理: 中間値を補完
    df = df.interpolate(method='linear')  # 線形補完（前日と後日の中間値）
    # 補完後も残るNaN（後日がない場合）は前日のデータで補完
    df = df.ffill()
    #print(df)
    df_stock.index = df_stock.index.tz_localize(None)  # タイムゾーン解除
    # stock_codeの日付と一致させる
    df = df.loc[df_stock.index]
    # 日付順に整列
    #df = df.sort_index()

    # 結果を保存
    dataframes_trend[ticker] = df
    dataframes_seasonal_0[ticker] = df
    dataframes_seasonal_1[ticker] = df
    dataframes_seasonal_2[ticker] = df
    dataframes_seasonal_3[ticker] = df
    dataframes_resid[ticker] = df
    
close_data = df_stock['Adj Close']  # pandas.Series
close_data = close_data.iloc[:, 0]
close_data = close_data.tz_localize(None)
print(close_data)

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
df_stock['SMA_200-50'] = df_stock['SMA_200'] - df_stock['SMA_50']

# 結果の確認
print(df_stock[['Volume', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'SMA_50', 'SMA_200', 'SMA_200-50']].tail())

tickers = [
    'Adj Close', 'Open',
    'CPI_Dates', 'CPIAUCSL', 'CPILFESL',
    'PCE_Dates', 'PCEPI', 'PCEPILFE',
    'PPI_Dates', 'PPIACO', 'WPSFD4131',
    'Unemployment_Rate_Dates', 'UNRATE',
    'Nonfarm_Payrolls_Release_Dates', 'PAYEMS',
    'Monetary_Base_Data_Dates', 'BOGMBASE'
] + fred_tickers + tech_indicator

#print(tickers)

for ticker in tech_indicator:
    if ticker == 'Volume':
        dataframes_trend[ticker] = df_stock[ticker].iloc[:, 0].tz_localize(None)
        dataframes_seasonal_0[ticker] = df_stock[ticker].iloc[:, 0].tz_localize(None)
        dataframes_seasonal_1[ticker] = df_stock[ticker].iloc[:, 0].tz_localize(None)
        dataframes_seasonal_2[ticker] = df_stock[ticker].iloc[:, 0].tz_localize(None)
        dataframes_seasonal_3[ticker] = df_stock[ticker].iloc[:, 0].tz_localize(None)
        dataframes_resid[ticker] = df_stock[ticker].iloc[:, 0].tz_localize(None)
    else:
        dataframes_trend[ticker] = df_stock[ticker]
        dataframes_seasonal_0[ticker] = df_stock[ticker]
        dataframes_seasonal_1[ticker] = df_stock[ticker]
        dataframes_seasonal_2[ticker] = df_stock[ticker]
        dataframes_seasonal_3[ticker] = df_stock[ticker]
        dataframes_resid[ticker] = df_stock[ticker]

# dataframes_trend の各データフレームから先頭200行を削除
#print("dataframes_trend:", dataframes_trend.items())
for ticker, df in dataframes_trend.items():
    dataframes_trend[ticker] = df.iloc[199:]  # 先頭200行を削除

for ticker, df in dataframes_seasonal_0.items():
    dataframes_seasonal_0[ticker] = df.iloc[199:]  # 先頭200行を削除

for ticker, df in dataframes_seasonal_1.items():
    dataframes_seasonal_1[ticker] = df.iloc[199:]  # 先頭200行を削除

for ticker, df in dataframes_seasonal_2.items():
    dataframes_seasonal_2[ticker] = df.iloc[199:]  # 先頭200行を削除

for ticker, df in dataframes_seasonal_3.items():
    dataframes_seasonal_3[ticker] = df.iloc[199:]  # 先頭200行を削除

for ticker, df in dataframes_resid.items():
    dataframes_resid[ticker] = df.iloc[199:]  # 先頭200行を削除

"""for ticker, df in dataframes_trend.items():
    print(f"{ticker} null values:\n{df.isnull().sum()}")"""

# ハイパーパラメータの取得
trend_params = best_hyperparams["2012-05-18_to_2023-06-01"]["trend"]["best_params"]
seasonal_params = {}
for i in range(4):
    seasonal_params[i] = best_hyperparams["2012-05-18_to_2023-06-01"][f"seasonal_{i}"][f"best_params_{i}"]
resid_params = best_hyperparams["2012-05-18_to_2023-06-01"]["resid"]["best_params"]
print(trend_params)
print(seasonal_params)
print(resid_params)
#exit()

# WandBの初期化
wandb.init(
    project="AMZN-stock-price-prediction(head = 4d)",
    name=f"{dataframes_trend['Adj_close'].index[0].strftime('%Y%m%d')}_{dataframes_trend['Adj_close'].index[-1].strftime('%Y%m%d')}[{start_date}_{end_date}]"
)

predict_period_num = 1
criterion = nn.MSELoss()
epochs =200

# トレーニングループ
models = {}
optimizers = {}
schedulers = {}
mean_lists = {}
std_lists = {}
train_data_dict = {}
valid_data_dict = {}
dataframes = {
    'trend': dataframes_trend,
    'seasonal_0': dataframes_seasonal_0,
    'seasonal_1': dataframes_seasonal_1,
    'seasonal_2': dataframes_seasonal_2,
    'seasonal_3': dataframes_seasonal_3,
    'resid': dataframes_resid
}

components = ['trend', 'seasonal_0', 'seasonal_1', 'seasonal_2', 'seasonal_3', 'resid']
for comp in components:
    if comp == 'trend':
        params = trend_params
    elif comp == 'resid':
        params = resid_params
    else:
        idx = int(comp.split('_')[1])
        params = seasonal_params[idx]
    #print(f"Training {comp} component", params)   

    # モデル、オプティマイザ、スケジューラの作成
    num_variates = len(tickers)
    model, optimizer, scheduler = create_model(params, len(tickers), predict_period_num, depth, dim)
    optimizers[comp] = optimizer
    schedulers[comp] = scheduler

    # データの正規化とデータセットの作成
    # 複数のデータフレームを結合
    combined_df = pd.DataFrame(dataframes[comp])

    # 正規化とデータセットの作成
    df_normalized, mean_list, std_list = data_Normalization(combined_df)

    mean_lists[comp] = mean_list
    std_lists[comp] = std_list
    train_data, valid_data = create_multivariate_dataset(
        df_normalized, params['observation_period_num'], predict_period_num, params['train_rates'], device)    
    train_data_dict[comp] = train_data
    valid_data_dict[comp] = valid_data
    
    # エポックごとのトレーニング
    earlystopping = EarlyStopping(patience=5)
    for epoch in range(epochs):
        models[comp], train_loss, valid_loss = train(model, train_data, valid_data, optimizer, criterion, scheduler, params['batch_size'], params['observation_period_num'])
        print(f"Epoch {epoch+1}/{epochs}, {comp} Loss: {train_loss:.4f} | {valid_loss:.4f}")
        # バリデーション損失の計算（適宜実装）
        earlystopping(valid_loss, model)
        if earlystopping.early_stop:
            print(f"Early stopping for {comp}")
            break
        scheduler.step()

        # WandBにログを記録
        wandb.log({f"{comp}_train_loss": train_loss, f"{comp}_valid_loss": valid_loss})

    # 学習済みモデルの保存
    torch.save(model.state_dict(), f'amzn_stock_price_{comp}_model.pth')


end_time = time.time()
runtime_seconds = end_time - start_time
print(f"Runtime (seconds): {runtime_seconds}")

# Update wandb configuration with hyperparameters
wandb.config.update({
    "learning_rate_trend": trend_params['learning_rate'],
    "learning_rate_seasonal_0": seasonal_params[0]['learning_rate'],
    "learning_rate_seasonal_1": seasonal_params[1]['learning_rate'],
    "learning_rate_seasonal_2": seasonal_params[2]['learning_rate'],
    "learning_rate_seasonal_3": seasonal_params[3]['learning_rate'],
    "learning_rate_resid": resid_params['learning_rate'],
    "epochs": epochs,
    "batch_size_trend": trend_params['batch_size'],
    "batch_size_seasonal_0": seasonal_params[0]['batch_size'],
    "batch_size_seasonal_1": seasonal_params[1]['batch_size'],
    "batch_size_seasonal_2": seasonal_params[2]['batch_size'],
    "batch_size_seasonal_3": seasonal_params[3]['batch_size'],
    "batch_size_resid": resid_params['batch_size'],
    "observation_period_trend": trend_params['observation_period_num'],
    "observation_period_seasonal_0": seasonal_params[0]['observation_period_num'],
    "observation_period_seasonal_1": seasonal_params[1]['observation_period_num'],
    "observation_period_seasonal_2": seasonal_params[2]['observation_period_num'],
    "observation_period_seasonal_3": seasonal_params[3]['observation_period_num'],
    "observation_period_resid": resid_params['observation_period_num'],
    "predict_period_num": predict_period_num
})

with torch.no_grad():
    # Prediction for trend, seasonal, and residual components
    last_batch_data_trend = valid_data_dict['trend'][-2][0].unsqueeze(0)
    last_batch_data_seasonal_0 = valid_data_dict['seasonal_0'][-2][0].unsqueeze(0)
    last_batch_data_seasonal_1 = valid_data_dict['seasonal_1'][-2][0].unsqueeze(0)
    last_batch_data_seasonal_2 = valid_data_dict['seasonal_2'][-2][0].unsqueeze(0)
    last_batch_data_seasonal_3 = valid_data_dict['seasonal_3'][-2][0].unsqueeze(0)
    last_batch_data_resid = valid_data_dict['resid'][-2][0].unsqueeze(0)

    predicted_trend = models['trend'](last_batch_data_trend)
    predicted_seasonal_0 = models['seasonal_0'](last_batch_data_seasonal_0)
    predicted_seasonal_1 = models['seasonal_1'](last_batch_data_seasonal_1)
    predicted_seasonal_2 = models['seasonal_2'](last_batch_data_seasonal_2)
    predicted_seasonal_3 = models['seasonal_3'](last_batch_data_seasonal_3)
    predicted_resid = models['resid'](last_batch_data_resid)

    # Select AMZN predictions
    predicted_trend_stock_price = predicted_trend[1][0, :, 0].cpu().numpy().flatten() * std_lists['trend'][0] + mean_lists['trend'][0]
    predicted_seasonal_0_stock_price = predicted_seasonal_0[1][0, :, 0].cpu().numpy().flatten() * std_lists['seasonal_0'][0] + mean_lists['seasonal_0'][0]
    predicted_seasonal_1_stock_price = predicted_seasonal_1[1][0, :, 0].cpu().numpy().flatten() * std_lists['seasonal_1'][0] + mean_lists['seasonal_1'][0]
    predicted_seasonal_2_stock_price = predicted_seasonal_2[1][0, :, 0].cpu().numpy().flatten() * std_lists['seasonal_2'][0] + mean_lists['seasonal_2'][0]
    predicted_seasonal_3_stock_price = predicted_seasonal_3[1][0, :, 0].cpu().numpy().flatten() * std_lists['seasonal_3'][0] + mean_lists['seasonal_3'][0]
    predicted_resid_stock_price = predicted_resid[1][0, :, 0].cpu().numpy().flatten() * std_lists['resid'][0] + mean_lists['resid'][0]


print(predicted_trend_stock_price)
print(predicted_seasonal_0_stock_price)
print(predicted_seasonal_1_stock_price)
print(predicted_seasonal_2_stock_price)
print(predicted_seasonal_3_stock_price)
print(predicted_resid_stock_price)

# Sum the components to get the final predicted stock price
final_predicted_stock_price = predicted_trend_stock_price + predicted_seasonal_0_stock_price + predicted_seasonal_1_stock_price + predicted_seasonal_2_stock_price + predicted_seasonal_3_stock_price + predicted_resid_stock_price
print(final_predicted_stock_price)
wandb.log({
    "real_trend_stock_price": stock_trend.iloc[-1],
    "real_seasonal_0_stock_price": stock_seasonal_0.iloc[-1],
    "real_seasonal_1_stock_price": stock_seasonal_1.iloc[-1],
    "real_seasonal_2_stock_price": stock_seasonal_2.iloc[-1],
    "real_seasonal_3_stock_price": stock_seasonal_3.iloc[-1],
    "real_resid_stock_price": stock_resid.iloc[-1],
    "predicted_trend_stock_price": predicted_trend_stock_price,
    "predicted_seasonal_0_stock_price": predicted_seasonal_0_stock_price,
    "predicted_seasonal_1_stock_price": predicted_seasonal_1_stock_price,
    "predicted_seasonal_2_stock_price": predicted_seasonal_2_stock_price,
    "predicted_seasonal_3_stock_price": predicted_seasonal_3_stock_price,
    "predicted_resid_stock_price": predicted_resid_stock_price,
    "final_predicted_stock_price": final_predicted_stock_price,
    "real_stock_price": close_data[-1]
})

output_date = 10
add_predicted_stock_price = np.append(close_data[-output_date:-predict_period_num].values, final_predicted_stock_price)
# Plot the final result
predicted_dates_tmp = close_data.index[-output_date:].strftime('%Y-%m-%d')
predicted_dates = predicted_dates_tmp.tolist()

plt.figure(figsize=(10, 6))
plt.plot(predicted_dates, close_data[-output_date:].values, linestyle='dashdot', color='green', label='Actual Price')
plt.plot(predicted_dates, add_predicted_stock_price, linestyle='dotted', color='red', label='Predicted Price')
plt.xlabel('Date', fontsize=16)
plt.ylabel('Stock Price', fontsize=16)
plt.legend(fontsize=14)
plt.title('AMZN Stock Price Prediction with Trend, Seasonal, and Residual Components', fontsize=18)

# Save and log to WandB
plt.savefig('amzn_stock_price_prediction_stl.png')
wandb.log({"AMZN Stock Price Prediction (STL)": wandb.Image('amzn_stock_price_prediction_stl.png')})

plt.show()


# Log runtime to WandB
wandb.log({"Runtime (seconds)": runtime_seconds})

# WandB finish
wandb.finish()
