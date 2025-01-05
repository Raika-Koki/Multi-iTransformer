import optuna
import pandas as pd
import json
import yfinance as yf
import time
import torch
import torch.nn as nn
import ta
import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.tseries.offsets import BDay
from scipy.stats import beta
from fredapi import Fred
from dotenv import load_dotenv
from src.Transformer_model import TransformerModel, EarlyStopping
from src.data_create import data_Normalization, create_multivariate_dataset
from src.train import train
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv("api_key.env")
fred_api_key = os.getenv("FRED_API")
fred = Fred(api_key=fred_api_key)

def save_best_hyperparameters(file_name, best_params, start_date, end_date):
    try:
        with open(file_name, 'r') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results = {}

    date_key = f"{start_date}_to_{end_date}"
    results[date_key] = best_params

    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)

def objective(trial, depth, dim):
    observation_period = trial.suggest_int('observation_period_num', 5, 252)
    train_rates = trial.suggest_float('train_rates', 0.6, 0.99)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 256)
    step_size = trial.suggest_int('step_size', 1, 15)
    gamma = trial.suggest_float('gamma', 0.75, 0.99)

    train_data, valid_data = create_multivariate_dataset(
        df_normalized, observation_period, predict_period_num, train_rates, device)
    model = TransformerModel(
        num_variates=len(tickers),
        lookback_len=observation_period,
        depth=depth,
        dim=dim,
        pred_length=predict_period_num
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.MSELoss()
    earlystopping = EarlyStopping(patience=5)

    for epoch in range(100): #check
        model, _, valid_loss = train(
            model, train_data, valid_data, optimizer, criterion, scheduler, batch_size, observation_period)
        earlystopping(valid_loss, model)
        if earlystopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    return valid_loss

def preprocess_open_data(data_open, reference_index):
    data_open.index = data_open.index - BDay(1)    
    data_open = data_open.dropna().reindex(reference_index).interpolate(method='linear').ffill()
    data_open.index = data_open.index.tz_localize(None)
    return data_open

def calculate_interest_levels(release_dates, date_range, alpha, beta_param, weight):
    interest_levels = pd.DataFrame({"Date": date_range, "Interest_Level": 0.0})
    for release_date in release_dates:
        day_offsets = (interest_levels["Date"] - release_date).dt.days
        within_range = day_offsets.between(-7, 2)
        x_values = (day_offsets[within_range] + 8) / 10
        if len(x_values) > 0:
            beta_pdf = beta.pdf(x_values, alpha, beta_param)
            interest_levels.loc[within_range, "Interest_Level"] += beta_pdf * weight
    interest_levels["Interest_Level"] /= interest_levels["Interest_Level"].max()
    return interest_levels

def create_model(num_variates, params, depth, dim, pred_length):
    model = TransformerModel(
        num_variates=num_variates,
        lookback_len=params['observation_period_num'],
        depth=depth,
        dim=dim,
        pred_length=pred_length
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    return model, optimizer, scheduler

# 初期設定
start_date = '2012-05-18'
initial_end_date = datetime.strptime('2024-11-25', '%Y-%m-%d')
stock_code = 'AAPL' #check
file_name = f"best_hyperparameters_{stock_code}_Transformer(nomstl).json"  # check
predict_period_num = 1
depth = 4
dim = 128

# 経済指標ごとに関心度を計算
alpha = 8
beta_param = 3
weights = {
    'CPI_Dates': 0.9,
    'PCE_Dates': 0.8,
    'PPI_Dates': 0.7,
    'GDP_Dates': 0.6,
    'Unemployment_Rate_Dates': 0.8,
    'Nonfarm_Payrolls_Release_Dates': 0.9,
    'Monetary_Base_Data_Dates': 0.6
}

# 1営業日ずつエンド日を伸ばして最適化を実施
while True:
    # 現在の end_date を文字列として設定
    end_date = initial_end_date.strftime('%Y-%m-%d')
    print(f"最適化対象のエンド日: {end_date}")

    # 現在の end_date でデータを取得
    df_stock = yf.download(stock_code, start=start_date, end=end_date)
    if len(df_stock) == 0:
        print(f"エンド日 {end_date} にデータが取得できません。処理を終了します。")
        break
    
    data_Adj_close = df_stock['Adj Close']
    data_Adj_close = data_Adj_close.tz_localize(None)
    data_Adj_close = data_Adj_close.iloc[:, 0]

    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    df_stock_dt = yf.download(stock_code, start=start_date_dt+BDay(1), end=end_date_dt+BDay(1))
    if len(df_stock_dt) == 0:
        print(f"エンド日 {end_date} にデータが取得できません。処理を終了します。")
        break

    # データの取得
    data_open = df_stock_dt['Open']
    data_open = preprocess_open_data(data_open, df_stock.index)
    data_open = data_open.tz_localize(None)
    data_open = data_open.iloc[:, 0]

    # データフレームの設定
    dataframes = {
        'Adj_close': data_Adj_close,
        'open': data_open
    }

    get_ticker = yf.Ticker(stock_code)
    splits = get_ticker.splits
    splits.index = splits.index.tz_localize(None)
    splits.index = splits.index - BDay(1)
    splits = splits.reindex(df_stock.tz_localize(None).index, fill_value=0)

    dataframes['splits'] = splits

    indicator_release_days = ['GDP_Dates', 'CPI_Dates', 'PCE_Dates', 'PPI_Dates', 'Unemployment_Rate_Dates', 'Nonfarm_Payrolls_Release_Dates', 'Monetary_Base_Data_Dates']#, 'Federal_Funds_Rate_Release_Dates']
    fred_tickers = ['DTWEXBGS', 'VIXCLS', 'DFII10', 'T10Y2Y']  # FRED tickers
    tech_indicator = ['Volume', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'SMA_50', 'SMA_200', 'SMA_200-50'] # Technical indicators

    # JSONファイルからデータを読み込む
    with open('Release_Dates.json', 'r') as file:
        release_dates_data = json.load(file)

    # 全ての日付範囲を生成
    date_range = pd.date_range(start=start_date, end=end_date)

    # 指標ごとに関心度を計算して出力
    for indicator in indicator_release_days:
        #print(indicator)
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
        dataframes[indicator] = interest_levels

        # FRED APIからデータを取得するためのマッピング
        indicator_to_fred_mapping = {
            'GDP_Dates': ['GDP'],
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

            #print(f'{fred_key}', indicator_data)
            
            # print(f'{fred_key}', indicator_data)
            # タイムゾーン解除
            indicator_data.index = indicator_data.index.tz_localize(None)
            # Reindex indicator_data to match df_stock's dates and forward fill values
            df_stock = df_stock.sort_index()
            indicator_data = indicator_data.sort_index()
            indicator_data = indicator_data.reindex(df_stock.index.tz_localize(None), method='ffill')
            
            #print(f'{fred_key}', indicator_data)

            # 結果を保存
            dataframes[f"{indicator}_{fred_key}"] = indicator_data


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

        #print(f'{ticker}', df)

        # 結果を保存
        dataframes[ticker] = df
        
    close_data = df_stock['Adj Close']  # pandas.Series
    close_data = close_data.iloc[:, 0]
    close_data = close_data.tz_localize(None)
    #print(close_data)

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
        'splits',
        'GDP_Dates', 'GDP',
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
            dataframes[ticker] = df_stock[ticker].iloc[:, 0].tz_localize(None)
        else:
            dataframes[ticker] = df_stock[ticker]

    # dataframes_trend の各データフレームから先頭200行を削除
    #print("dataframes_trend:", dataframes_trend.items())
    for ticker, df in dataframes.items():
        dataframes[ticker] = df.iloc[199:]  # 先頭200行を削除


    """for ticker, df in dataframes_trend.items():
        print(f"{ticker} null values:\n{df.isnull().sum()}")"""
    
    print(dataframes)
    # 複数のデータフレームを結合
    combined_df = pd.DataFrame(dataframes)

    # 正規化とデータセットの作成
    df_normalized, mean_list, std_list = data_Normalization(combined_df)

    # WandBの初期化
    wandb.init(
        project=f"{stock_code}-stock-price-prediction-notMSTL-by-Transformer",
        name=f"{combined_df.index[0].strftime('%Y%m%d')}_{combined_df.index[-1].strftime('%Y%m%d')}[{start_date}_{end_date}]"
    )

    # 最適化と結果の保存
    best_params = {}  # Updated structure

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, depth, dim), n_trials=50) #check
    if len(study.trials) > 0 and any([t.state == optuna.trial.TrialState.COMPLETE for t in study.trials]):
        best_params = study.best_params
        print("最適ハイパーパラメータが見つかりました")
    else:
        print("No completed trials. Skipping.")


    # 最適なハイパーパラメータと日付範囲をファイルに保存
    save_best_hyperparameters(file_name, best_params, start_date, end_date)
    print(f"最適ハイパーパラメータが {file_name} に保存されました")

    predict_period_num = 1
    criterion = nn.MSELoss()
    epochs = 300 #check

    # 学習時間の計測
    start_time = time.time()

    # トレーニングループ
    model, optimizer, scheduler = create_model(len(tickers), best_params, depth, dim, predict_period_num)

    # データの正規化とデータセットの作成
    train_data, valid_data = create_multivariate_dataset(
        df_normalized, best_params['observation_period_num'], predict_period_num, best_params['train_rates'], device)

    # エポックごとのトレーニング
    earlystopping = EarlyStopping(patience=5)
    for epoch in range(epochs):
        model, train_loss, valid_loss = train(
            model, train_data, valid_data, optimizer, criterion, scheduler,
            best_params['batch_size'], best_params['observation_period_num'])
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f} | {valid_loss:.4f}")
        earlystopping(valid_loss, model)
        if earlystopping.early_stop:
            print(f"Early stopping")
            break
        scheduler.step()

        # WandBにログを記録
        wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})

    # 学習済みモデルの保存
    torch.save(model.state_dict(), f'{stock_code}_stock_price_model.pth')

    end_time = time.time()

    runtime_seconds = end_time - start_time
    print(f"Runtime (seconds): {runtime_seconds}")

    # Update wandb configuration with hyperparameters
    config_updates = {
        "learning_rate": best_params.get('learning_rate'),
        "batch_size": best_params.get('batch_size'),
        "observation_period_num": best_params.get('observation_period_num'),
        "epochs": epochs,
        "predict_period_num": predict_period_num,
        "depth": depth,
        "dim": dim
    }
    wandb.config.update(config_updates)

    # Prediction部分を修正
    with torch.no_grad():
        # 最後のバッチデータを取得
        last_batch_data = valid_data[-2][0].unsqueeze(0)

        # 予測
        predicted = model(last_batch_data)

        predicted_stock_price = predicted[0, :, 0].cpu().numpy().flatten()  * std_list.iloc[0] + mean_list.iloc[0]

    actual_stock_price = close_data[-predict_period_num:].values  # Actual stock price
    # Calculate MSE, RMSE, MAE, and R-squared
    mse = mean_squared_error(actual_stock_price, predicted_stock_price)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_stock_price, predicted_stock_price)
    r2 = r2_score(actual_stock_price, predicted_stock_price)

    # 結果を出力
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R-squared: {r2}")
    print(predicted_stock_price)

    wandb.log({
        "predicted_stock_price": predicted_stock_price,
        "real_stock_price": actual_stock_price,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "depth": depth,
        "dim": dim
    })

    output_date = 10
    add_predicted_stock_price = np.append(close_data[-output_date:-predict_period_num].values, predicted_stock_price)
    # Plot the final result
    predicted_dates_tmp = close_data.index[-output_date:].strftime('%Y-%m-%d')
    predicted_dates = predicted_dates_tmp.tolist()
    learning_dates_tmp = close_data.index[-output_date:-1].strftime('%Y-%m-%d')
    learning_dates = learning_dates_tmp.tolist()

    plt.figure(figsize=(10, 6))
    plt.plot(predicted_dates, close_data[-output_date:].values, linestyle='dashdot', color='green', label='Actual Price')
    plt.plot(predicted_dates, add_predicted_stock_price, linestyle='dotted', color='red', label='Predicted Price')
    plt.plot(learning_dates, close_data[-output_date:-1].values, color='black', label='learning data')
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Stock Price', fontsize=16)
    plt.legend(fontsize=14)
    plt.title(f'{stock_code} Stock Price Prediction by Transformer(No MSTL)', fontsize=18) #check

    # 保存するファイル名を一致させる
    plt.savefig(f'{stock_code.lower()}_stock_price_prediction_by_Transformer(nomstl).png') # 修正
    # WandBにログを送信
    wandb.log({
        f"{stock_code} Stock Price Prediction by Transformer(no MSTL)": wandb.Image(f'{stock_code.lower()}_stock_price_prediction_by_Transformer(nomstl).png') # 修正
    })

    plt.show()

    # Log runtime to WandB
    wandb.log({"Runtime (seconds)": runtime_seconds})

    # WandB finish
    wandb.finish()

    # 次の営業日に移行
    initial_end_date += BDay(1)  # 1営業日進める

    # 他のプロセスが中間結果を確認できるようにする
    print("次のエンド日に進む前に一時停止します...")
    time.sleep(10)  # 必要に応じて調整
