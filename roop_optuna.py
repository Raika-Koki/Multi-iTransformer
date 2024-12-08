import optuna
import pandas as pd
import json
import yfinance as yf
import time
import torch
import torch.nn as nn
import ta
import os
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import BDay  # 1営業日単位で増加
from scipy.stats import beta
from fredapi import Fred
from dotenv import load_dotenv
from src.model import iTransformer, EarlyStopping
from src.data_create import data_Normalization, create_multivariate_dataset
from src.train import train
from statsmodels.tsa.seasonal import MSTL

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# .envファイルからAPIキーを読み込む
load_dotenv("api_key.env")
fred_api_key = os.getenv("FRED_API")
print(f"API Key: {fred_api_key}")  # 確認用の出力
# APIキーを渡してFREDクライアントを作成
fred = Fred(api_key=fred_api_key)

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
        "seasonal_0": {
            "best_params_0": best_params["seasonal_0"]
        },
        "seasonal_1": {
            "best_params_1": best_params["seasonal_1"]
        },
        "seasonal_2": {
            "best_params_2": best_params["seasonal_2"]
        },
        "seasonal_3": {
            "best_params_3": best_params["seasonal_3"]
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
    dataset_mapping = {
        "trend": df_normalized_trend,
        "seasonal_0": df_normalized_seasonal_0,
        "seasonal_1": df_normalized_seasonal_1,
        "seasonal_2": df_normalized_seasonal_2,
        "seasonal_3": df_normalized_seasonal_3,
        "resid": df_normalized_resid
    }

    train_data, valid_data = create_multivariate_dataset(
        dataset_mapping[component], observation_period, predict_period_num, train_rates, device)
    model = iTransformer(
        num_variates=len(tickers),
        lookback_len=observation_period,
        depth=depth,
        dim=dim,
        pred_length=predict_period_num
    ).to(device)

    # オプティマイザとスケジューラの定義
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.MSELoss()
    earlystopping = EarlyStopping(patience=5)

    for _ in range(5):  # 'epoch' を使用しないため '_' に変更
        model, _, valid_loss = train(
            model, train_data, valid_data, optimizer, criterion, scheduler, batch_size, observation_period)
        earlystopping(valid_loss, model)
        if earlystopping.early_stop:
            print("Early stopping")
            break

    return valid_loss

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

# 初期設定
start_date = '2012-05-18'
initial_end_date = datetime.strptime('2023-06-01', '%Y-%m-%d')
file_name = "best_hyperparameters_AAPL_1.json"
predict_period_num = 1

#beta関数のパラメータ設定
alpha = 8
beta_param = 3
weight = 0.5

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
    
    data_Adj_close = df_stock['Adj Close']
    #print(data_Adj_close)

    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    #print(f"start_date_dt: {start_date_dt}, end_date_dt: {end_date_dt}")
    df_stock_dt = yf.download(stock_code, start=start_date_dt+BDay(1), end=end_date_dt+BDay(1))
    if len(df_stock_dt) == 0:
        print(f"エンド日 {end_date} にデータが取得できません。処理を終了します。")
        break
    

    # データの取得
    data_open = df_stock_dt['Open']
    data_open = preprocess_open_data(data_open, df_stock.index)

    # MSTLによる分解
    periods = [252, 504, 756, 1260]
    iterate = 3
    stl_kwargs = {"seasonal_deg": 0, "inner_iter": 3, "outer_iter": 1}

    mstl_series = decompose_stock_data(data_Adj_close, periods, iterate, stl_kwargs)
    mstl_series_open = decompose_stock_data(data_open, periods, iterate, stl_kwargs)

    #print(f"Seasonal Component ({periods[3]} days):")
    #print(mstl_series.seasonal.iloc[:, 3])  # .ilocで列を指定

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

    #print(f"aaa", dataframes_trend)


    indicator_release_days = ['CPI', 'PCE', 'PPI', 'Unemployment_Rate', 'Nonfarm_Payrolls_Release_Dates', 'Monetary_Base_Data', 'Federal_Funds_Rate_Release_Dates']
    fred_tickers = ['DTWEXBGS', 'VIXCLS', 'DFII10', 'T10Y2Y']  # FRED tickers
    tech_indicator = ['Volume', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'SMA_50', 'SMA_200'] # Technical indicators

    # JSONファイルからデータを読み込む
    with open('Release_Dates.json', 'r') as file:
        release_dates_data = json.load(file)

    # 全ての日付範囲を生成
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

        release_dates = pd.to_datetime(release_dates)
        interest_levels = calculate_interest_levels(release_dates, date_range, alpha, beta_param, weight)

        # 'Date'列を日付型に変換してインデックスに設定
        interest_levels['Date'] = pd.to_datetime(interest_levels['Date'])
        interest_levels.set_index('Date', inplace=True)
        
        # interest_levels のデータをフィルタリングして df_stock の日付に一致するものだけ残す
        interest_levels = interest_levels.loc[interest_levels.index.intersection(df_stock.index.tz_localize(None))]
        interest_levels = interest_levels.iloc[:, 0]  # 1列目のみ残す
        #print(interest_levels.shape)

        """# 結果を表示
        non_zero_interest = interest_levels[interest_levels["Interest_Level"] > 0]
        print(f"Interest levels for {indicator}:")
        print(non_zero_interest.head(20))  # 関心度が非ゼロの日付のみ表示"""

        # 結果を保存
        dataframes_trend[indicator] = interest_levels
        dataframes_seasonal_0[indicator] = interest_levels
        dataframes_seasonal_1[indicator] = interest_levels
        dataframes_seasonal_2[indicator] = interest_levels
        dataframes_seasonal_3[indicator] = interest_levels
        dataframes_resid[indicator] = interest_levels


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

    # 結果の確認
    print(df_stock[['Volume', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'SMA_50', 'SMA_200']].tail())

    tickers = [stock_code] + ['open'] + indicator_release_days + fred_tickers + tech_indicator 

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
    

    # 複数のデータフレームを結合
    combined_df_trend = pd.DataFrame(dataframes_trend)
    combined_df_seasonal_0 = pd.DataFrame(dataframes_seasonal_0)
    combined_df_seasonal_1 = pd.DataFrame(dataframes_seasonal_1)
    combined_df_seasonal_2 = pd.DataFrame(dataframes_seasonal_2)
    combined_df_seasonal_3 = pd.DataFrame(dataframes_seasonal_3)
    combined_df_resid = pd.DataFrame(dataframes_resid)

    # 正規化とデータセットの作成
    df_normalized_trend, mean_list_trend, std_list_trend = data_Normalization(combined_df_trend)
    df_normalized_seasonal_0, mean_list_seasonal_0, std_list_seasonal_0 = data_Normalization(combined_df_seasonal_0)
    df_normalized_seasonal_1, mean_list_seasonal_1, std_list_seasonal_1 = data_Normalization(combined_df_seasonal_1)    
    df_normalized_seasonal_2, mean_list_seasonal_2, std_list_seasonal_2 = data_Normalization(combined_df_seasonal_2)
    df_normalized_seasonal_3, mean_list_seasonal_3, std_list_seasonal_3 = data_Normalization(combined_df_seasonal_3)
    df_normalized_resid, mean_list_resid, std_list_resid = data_Normalization(combined_df_resid)

    # 最適化と結果の保存
    best_params = { "trend": {}, "seasonal_0": {}, "seasonal_1": {}, "seasonal_2": {}, "seasonal_3": {}, "resid": {} }  # ここで毎回初期化

    for component, study_name in zip(["trend", "seasonal_0", "seasonal_1", "seasonal_2", "seasonal_3", "resid"], ["study_trend", "study_seasonal_0", "study_seasonal_1", "study_seasonal_2", "study_seasonal_3", "study_resid"]):
        print(f"最適化対象: {component}")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, component), n_trials=1) #ここを変えた

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
