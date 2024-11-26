import optuna
import pandas as pd
import json
from datetime import datetime
from pandas.tseries.offsets import BDay  # 1営業日単位で増加
import yfinance as yf
from statsmodels.tsa.seasonal import STL  # STL分解
import time
import torch
import torch.nn as nn
import numpy as np
import ta
import os
from fredapi import Fred
from dotenv import load_dotenv
from src.model import iTransformer, EarlyStopping
from src.data_create import data_Normalization, create_multivariate_dataset
from src.train import train

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# .envファイルからAPIキーを読み込む
load_dotenv("fred_api.env")
api_key = os.getenv("API_KEY")
print(f"API Key: {api_key}")  # 確認用の出力
# APIキーを渡してFREDクライアントを作成
fred = Fred(api_key=api_key)

# ハイパーパラメータを JSON ファイルに保存する関数
def save_best_hyperparameters(file_name, best_params, start_date, end_date):
    try:
        with open(file_name, 'r') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results = {}

    # 日付をキーにしてその日付ごとの結果を保存
    date_key = f"{start_date}_to_{end_date}"
    results[date_key] = {
        "dates": {
            "start_date": start_date,
            "end_date": end_date
        },
        "trend": {
            "best_params": best_params["trend"]
        },
        "seasonal": {
            "best_params": best_params["seasonal"]
        },
        "resid": {
            "best_params": best_params["resid"]
        }
    }

    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)


# Optuna での最適化対象の関数
def objective(trial, component):
    # ハイパーパラメータの提案
    observation_period = trial.suggest_int('observation_period_num', 5, 252)
    train_rates = trial.suggest_float('train_rates', 0.6, 0.99)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 256)
    step_size = trial.suggest_int('step_size', 1, 15)
    gamma = trial.suggest_float('gamma', 0.75, 0.99)
    depth = trial.suggest_int('depth', 2, 6)
    dim = trial.suggest_int('dim', 16, 256)

    # データとモデルの設定
    if component == "trend":
        #print(tickers)
        train_data, valid_data = create_multivariate_dataset(
            df_normalized_trend, observation_period, predict_period_num, train_rates, device)
        model = iTransformer(num_variates=len(tickers), lookback_len=observation_period, depth=depth, dim=dim, pred_length=predict_period_num).to(device)
    elif component == "seasonal":
        train_data, valid_data = create_multivariate_dataset(
            df_normalized_seasonal, observation_period, predict_period_num, train_rates, device)
        model = iTransformer(num_variates=len(tickers), lookback_len=observation_period, depth=depth, dim=dim, pred_length=predict_period_num).to(device)
    else:
        train_data, valid_data = create_multivariate_dataset(
            df_normalized_resid, observation_period, predict_period_num, train_rates, device)
        model = iTransformer(num_variates=len(tickers), lookback_len=observation_period, depth=depth, dim=dim, pred_length=predict_period_num).to(device)

    # オプティマイザとスケジューラの定義
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.MSELoss()
    earlystopping = EarlyStopping(patience=5)

    # モデルのトレーニング
    for epoch in range(100):
        model, train_loss, valid_loss = train(
            model, train_data, valid_data, optimizer, criterion, scheduler, batch_size, observation_period)
        earlystopping(valid_loss, model)
        if earlystopping.early_stop:
            print("Early stopping")
            break

    return valid_loss

# 初期設定
start_date = '2012-05-18'
initial_end_date = datetime.strptime('2023-06-01', '%Y-%m-%d')
file_name = "best_hyperparameters_AAPL.json"
predict_period_num = 1

# 1営業日ずつエンド日を伸ばして最適化を実施
while True:
    stock_code = 'AAPL'

    # 現在の end_date を文字列として設定
    end_date = initial_end_date.strftime('%Y-%m-%d')
    print(f"最適化対象のエンド日: {end_date}")

    # 現在の end_date でデータを取得し、STL 分解を実行
    df_stock = yf.download(stock_code, start=start_date, end=end_date)
    if len(df_stock) == 0:
        print(f"エンド日 {end_date} にデータが取得できません。処理を終了します。")
        break

    data = df_stock['Adj Close']
    #print(data)

    stl = STL(data, period=252, robust=True)
    stl_series = stl.fit()

    # トレンド、季節性、残差の成分を抽出
    stock_trend = stl_series.trend.tz_localize(None)
    stock_seasonal = stl_series.seasonal.tz_localize(None)
    stock_resid = stl_series.resid.tz_localize(None)

    # 現在の end_date に基づいたデータフレームを更新
    dataframes_trend = {stock_code: stock_trend}
    dataframes_seasonal = {stock_code: stock_seasonal}
    dataframes_resid = {stock_code: stock_resid}

    fred_tickers = ['DTWEXBGS', 'VIXCLS', 'DFII10', 'T10Y2Y']  # FRED tickers
    tech_indicator = ['Volume', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'SMA_50', 'SMA_200'] # Technical indicators


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
        print(df)
        df_stock.index = df_stock.index.tz_localize(None)  # タイムゾーン解除
        # stock_codeの日付と一致させる
        df = df.loc[df_stock.index]
        # 日付順に整列
        #df = df.sort_index()

        # 結果を保存
        dataframes_trend[ticker] = df
        dataframes_seasonal[ticker] = df
        dataframes_resid[ticker] = df
        
    close_data = df_stock['Adj Close']  # pandas.Series
    close_data = data.iloc[:, 0]
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

    # 結果の確認
    print(df_stock[['Volume', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'SMA_50', 'SMA_200']].tail())

    tickers = [stock_code] + fred_tickers + tech_indicator

    #print(tickers)

    for ticker in tech_indicator:
        if ticker == 'Volume':
            dataframes_trend[ticker] = df_stock[ticker].iloc[:, 0].tz_localize(None)
            dataframes_seasonal[ticker] = df_stock[ticker].iloc[:, 0].tz_localize(None)
            dataframes_resid[ticker] = df_stock[ticker].iloc[:, 0].tz_localize(None)
        else:
            dataframes_trend[ticker] = df_stock[ticker]
            dataframes_seasonal[ticker] = df_stock[ticker]
            dataframes_resid[ticker] = df_stock[ticker]

    # dataframes_trend の各データフレームから先頭200行を削除
    #print("dataframes_trend:", dataframes_trend.items())
    for ticker, df in dataframes_trend.items():
        dataframes_trend[ticker] = df.iloc[199:]  # 先頭200行を削除

    for ticker, df in dataframes_seasonal.items():
        dataframes_seasonal[ticker] = df.iloc[199:]  # 先頭200行を削除

    for ticker, df in dataframes_resid.items():
        dataframes_resid[ticker] = df.iloc[199:]  # 先頭200行を削除

    for ticker, df in dataframes_trend.items():
        print(f"{ticker} null values:\n{df.isnull().sum()}")

    print(dataframes_trend)

    # 複数のデータフレームを結合
    combined_df_trend = pd.DataFrame(dataframes_trend)
    combined_df_seasonal = pd.DataFrame(dataframes_seasonal)
    combined_df_resid = pd.DataFrame(dataframes_resid)

    # 正規化とデータセットの作成
    df_normalized_trend, mean_list_trend, std_list_trend = data_Normalization(combined_df_trend)
    df_normalized_seasonal, mean_list_seasonal, std_list_seasonal = data_Normalization(combined_df_seasonal)
    df_normalized_resid, mean_list_resid, std_list_resid = data_Normalization(combined_df_resid)

    # 最適化と結果の保存
    best_params = { "trend": {}, "seasonal": {}, "resid": {} }  # ここで毎回初期化

    for component, study_name in zip(["trend", "seasonal", "resid"], ["study_trend", "study_seasonal", "study_resid"]):
        print(f"最適化対象: {component}")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, component), n_trials=100)

        # 最適なハイパーパラメータを辞書に保存
        best_params[component] = study.best_params

        print(f"{component} の最適ハイパーパラメータが見つかりました")

    # 最適なハイパーパラメータと日付範囲をファイルに保存
    save_best_hyperparameters(file_name, best_params, start_date, end_date)
    print(f"最適ハイパーパラメータが {file_name} に保存されました")

    # 次の営業日に移行
    initial_end_date += BDay(1)  # 1営業日進める

    # 他のプロセスが中間結果を確認できるようにする
    print("次のエンド日に進む前に一時停止します...")
    time.sleep(10)  # 必要に応じて調整
