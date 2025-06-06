import optuna
import pandas as pd
import torch
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
from src.model import iTransformer, EarlyStopping
from src.data_create import data_Normalization, create_multivariate_dataset
from src.train import train

# WandB initialization
wandb.init(project="gafam-stock-price-prediction")

# Start time
start_time = time.time()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 確認用の関数
def check_device(tensor):
    if tensor.device == torch.device("cuda:0"):
        print("Tensor is on GPU.")
    else:
        print("Tensor is on CPU.")

# .envファイルからAPIキーを読み込む
load_dotenv("fred_api.env")
api_key = os.getenv("API_KEY")
#print(f"API Key: {api_key}")  # 確認用の出力
# APIキーを渡してFREDクライアントを作成
fred = Fred(api_key=api_key)

# Download stock data
stock_code = 'AAPL'
start_date = '2012-05-18'
end_date = '2023-06-01'

df_stock = yf.download(stock_code, start=start_date, end=end_date)
if len(df_stock) == 0:
    raise Exception("No data fetched for the given stock code and date range.")

df_stock.index = df_stock.index.tz_localize(None)

data = df_stock['Adj Close']
print(data)

# stock STL decomposition to stock data
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
    # 再インデックスで欠損日付を挿入（NaNで埋まる）
    df = df.reindex(full_date_range)
    # 補完処理: 中間値を補完
    df = df.interpolate(method='linear')  # 線形補完（前日と後日の中間値）
    # 補完後も残るNaN（後日がない場合）は前日のデータで補完
    df = df.fillna(method='ffill')  # 前日のデータを使用して補完
    # stock_codeの日付と一致させる
    df = df.loc[df_stock.index]
    # 日付順に整列
    df = df.sort_index()

    # 結果を保存
    dataframes_trend[ticker] = df
    dataframes_seasonal[ticker] = df
    dataframes_resid[ticker] = df

#print(dataframes_trend)

"""# AAPLの日付が他のデータ1にない場合
missing_in_other_1 = df_stock.index.difference(dataframes_trend['DTWEXBGS'].index)
print("AAPLにあるが他のデータ1にはない日付:", missing_in_other_1)

# 他のデータ1の日付がAAPLにない場合
missing_in_aapl = dataframes_trend['DTWEXBGS'].index.difference(df_stock.index)
print("他のデータ1にあるがAAPLにはない日付:", missing_in_aapl)"""

close_data = data.iloc[:, 0]

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
print(df_stock[['BB_Upper', 'BB_Lower', 'BB_Middle', 'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'SMA_50', 'SMA_200']].tail())

tickers = [stock_code] + fred_tickers + tech_indicator

#print(tickers)

for ticker in tech_indicator:
    if ticker == 'Volume':
        dataframes_trend[ticker] = df_stock[ticker].iloc[:, 0]
        dataframes_seasonal[ticker] = df_stock[ticker].iloc[:, 0]
        dataframes_resid[ticker] = df_stock[ticker].iloc[:, 0]
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

#print("dataframes_trend:", dataframes_trend.items())

"""print(dataframes_trend.items())
print(dataframes_seasonal.items())
print(dataframes_resid.items())
"""

"""print(dataframes_trend)

missing_in_other_1 = dataframes_trend['AAPL'].index.difference(dataframes_trend['SMA_200'].index)
print("AAPLにあるが他のデータ1にはない日付:", missing_in_other_1)
"""
for ticker, df in dataframes_trend.items():
    print(f"{ticker} null values:\n{df.isnull().sum()}")

# Combine the stock data into a single DataFrame
combined_df_trend = pd.DataFrame(dataframes_trend)
combined_df_seasonal = pd.DataFrame(dataframes_seasonal)
combined_df_resid = pd.DataFrame(dataframes_resid)

# Normalize the full dataset (GAFAM stock data)
df_normalized_trend, mean_list_trend, std_list_trend = data_Normalization(combined_df_trend)
df_normalized_seasonal, mean_list_seasonal, std_list_seasonal = data_Normalization(combined_df_seasonal)
df_normalized_resid, mean_list_resid, std_list_resid = data_Normalization(combined_df_resid)

# Define STL mean and std lists
stl_mean_list = [mean_list_trend[stock_code], mean_list_seasonal[stock_code], mean_list_resid[stock_code]]
stl_std_list = [std_list_trend[stock_code], std_list_seasonal[stock_code], std_list_resid[stock_code]]

# Create multivariate dataset for each component
predict_period_num = 1
epochs = 100

# Optuna objective function for hyperparameter tuning
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
    for epoch in range(1):
        model, train_loss, valid_loss = train(
            model, train_data, valid_data, optimizer, criterion, scheduler, batch_size, observation_period)
        earlystopping(valid_loss, model)
        if earlystopping.early_stop:
            print("Early stopping")
            break

    return valid_loss

# Optimize for each component
study_trend = optuna.create_study(direction='minimize')
study_trend.optimize(lambda trial: objective(trial, "trend"), n_trials=1)

study_seasonal = optuna.create_study(direction='minimize')
study_seasonal.optimize(lambda trial: objective(trial, "seasonal"), n_trials=1)

study_resid = optuna.create_study(direction='minimize')
study_resid.optimize(lambda trial: objective(trial, "resid"), n_trials=1)

# Log best hyperparameters
print("Best hyperparameters (trend):", study_trend.best_params)
print("Best hyperparameters (seasonal):", study_seasonal.best_params)
print("Best hyperparameters (resid):", study_resid.best_params)

# Extract best hyperparameters from Optuna studies for each component
best_params_trend = study_trend.best_params
best_params_seasonal = study_seasonal.best_params
best_params_resid = study_resid.best_params

# Initialize separate models for trend, seasonal, and residual components using optimized parameters
model_trend = iTransformer(
    num_variates=len(tickers),
    lookback_len=best_params_trend['observation_period_num'],
    depth=best_params_trend['depth'],
    dim=best_params_trend['dim'],
    pred_length=predict_period_num,
).to(device)

model_seasonal = iTransformer(
    num_variates=len(tickers),
    lookback_len=best_params_seasonal['observation_period_num'],
    depth=best_params_seasonal['depth'],
    dim=best_params_seasonal['dim'],
    pred_length=predict_period_num
    ).to(device)

model_resid = iTransformer(
    num_variates=len(tickers),
    lookback_len=best_params_resid['observation_period_num'],
    depth=best_params_resid['depth'],
    dim=best_params_resid['dim'],
    pred_length=predict_period_num
).to(device)

# Define loss function
criterion = nn.MSELoss()

# Define separate optimizers and schedulers for each model based on optimized parameters
optimizer_trend = torch.optim.AdamW(model_trend.parameters(), lr=best_params_trend['learning_rate'])
scheduler_trend = torch.optim.lr_scheduler.StepLR(optimizer_trend, step_size=best_params_trend['step_size'], gamma=best_params_trend['gamma'])

optimizer_seasonal = torch.optim.AdamW(model_seasonal.parameters(), lr=best_params_seasonal['learning_rate'])
scheduler_seasonal = torch.optim.lr_scheduler.StepLR(optimizer_seasonal, step_size=best_params_seasonal['step_size'], gamma=best_params_seasonal['gamma'])

optimizer_resid = torch.optim.AdamW(model_resid.parameters(), lr=best_params_resid['learning_rate'])
scheduler_resid = torch.optim.lr_scheduler.StepLR(optimizer_resid, step_size=best_params_resid['step_size'], gamma=best_params_resid['gamma'])

earlystopping = EarlyStopping(patience=5)

# Log hyperparameters to WandB
wandb.config.update({
    "learning_rate_trend": best_params_trend['learning_rate'],
    "learning_rate_seasonal": best_params_seasonal['learning_rate'],
    "learning_rate_resid": best_params_resid['learning_rate'],
    "epochs": epochs,
    "batch_size_trend": best_params_trend['batch_size'],
    "batch_size_seasonal": best_params_seasonal['batch_size'],
    "batch_size_resid": best_params_resid['batch_size'],
    "observation_period_num": best_params_trend['observation_period_num'],
    "observation_period_num": best_params_seasonal['observation_period_num'],
    "observation_period_num": best_params_resid['observation_period_num'],
    "predict_period_num": predict_period_num
})

# Create datasets for each component (trend, seasonal, residual)
train_data_trend, valid_data_trend = create_multivariate_dataset(
    df_normalized_trend, best_params_trend['observation_period_num'], predict_period_num, best_params_trend['train_rates'], device)
train_data_seasonal, valid_data_seasonal = create_multivariate_dataset(
    df_normalized_seasonal, best_params_seasonal['observation_period_num'], predict_period_num, best_params_seasonal['train_rates'], device)
train_data_resid, valid_data_resid = create_multivariate_dataset(
    df_normalized_resid, best_params_resid['observation_period_num'], predict_period_num, best_params_resid['train_rates'], device)

# Training loop
for epoch in range(epochs):
    # Train trend model
    model_trend, train_loss_trend, valid_loss_trend = train(
        model_trend, train_data_trend, valid_data_trend, optimizer_trend, criterion, scheduler_trend, best_params_trend['batch_size'], best_params_trend['observation_period_num'])
    
    # Train seasonal model
    model_seasonal, train_loss_seasonal, valid_loss_seasonal = train(
        model_seasonal, train_data_seasonal, valid_data_seasonal, optimizer_seasonal, criterion, scheduler_seasonal, best_params_seasonal['batch_size'], best_params_seasonal['observation_period_num'])
    
    # Train residual model
    model_resid, train_loss_resid, valid_loss_resid = train(
        model_resid, train_data_resid, valid_data_resid, optimizer_resid, criterion, scheduler_resid, best_params_resid['batch_size'], best_params_resid['observation_period_num'])

    print(f"Epoch {epoch+1}/{epochs}, (Training | Validation) Trend Loss: {train_loss_trend:.4f} | {valid_loss_trend:.4f}, "
      f"Seasonal Loss: {train_loss_seasonal:.4f} | {valid_loss_seasonal:.4f}, "
      f"Residual Loss: {train_loss_resid:.4f} | {valid_loss_resid:.4f}")

    
    # Log losses to WandB
    wandb.log({
        "Train Loss (Trend)": train_loss_trend, "Validation Loss (Trend)": valid_loss_trend,
        "Train Loss (Seasonal)": train_loss_seasonal, "Validation Loss (Seasonal)": valid_loss_seasonal,
        "Train Loss (Resid)": train_loss_resid, "Validation Loss (Resid)": valid_loss_resid
    })

    # Early stopping based on residual validation loss
    earlystopping(valid_loss_resid, model_resid)
    if earlystopping.early_stop:
        print("Early stopping")
        break

# Prediction on the validation set
model_trend.eval()
model_seasonal.eval()
model_resid.eval()

# Save the models using torch.save
torch.save(model_trend.state_dict(), 'gafam_stock_price_trend_model.pth')
torch.save(model_seasonal.state_dict(), 'gafam_stock_price_seasonal_model.pth')
torch.save(model_resid.state_dict(), 'gafam_stock_price_resid_model.pth')

with torch.no_grad():
    # Prediction for trend, seasonal, and residual components
    last_batch_data_trend = valid_data_trend[-2][0].unsqueeze(0)
    last_batch_data_seasonal = valid_data_seasonal[-2][0].unsqueeze(0)
    last_batch_data_resid = valid_data_resid[-2][0].unsqueeze(0)

    predicted_trend = model_trend(last_batch_data_trend)
    predicted_seasonal = model_seasonal(last_batch_data_seasonal)
    predicted_resid = model_resid(last_batch_data_resid)


    # Select AAPL predictions
    predicted_trend_stock_price = predicted_trend[1][0, :, 0].cpu().numpy().flatten() * stl_std_list[0] + stl_mean_list[0]
    predicted_seasonal_stock_price = predicted_seasonal[1][0, :, 0].cpu().numpy().flatten() * stl_std_list[1] + stl_mean_list[1]
    predicted_resid_stock_price = predicted_resid[1][0, :, 0].cpu().numpy().flatten() * stl_std_list[2] + stl_mean_list[2]

print(predicted_trend_stock_price)
print(predicted_seasonal_stock_price)
print(predicted_resid_stock_price)

# Sum the components to get the final predicted stock price
final_predicted_stock_price = predicted_trend_stock_price + predicted_seasonal_stock_price + predicted_resid_stock_price
print(final_predicted_stock_price)
wandb.log({
    "real_trend_stock_price": stock_trend[-1],
    "real_seasonal_stock_price": stock_seasonal[-1],
    "real_resid_stock_price": stock_resid[-1],
    "predicted_trend_stock_price": predicted_trend_stock_price,
    "predicted_seasonal_stock_price": predicted_seasonal_stock_price,
    "predicted_resid_stock_price":
      predicted_resid_stock_price,
    "final_predicted_stock_price": final_predicted_stock_price
})

output_date = 10
add_predicted_stock_price = np.append(data[-output_date:-predict_period_num].values, final_predicted_stock_price)
# Plot the final result
predicted_dates_tmp = data.index[-output_date:].strftime('%Y-%m-%d')
predicted_dates = predicted_dates_tmp.tolist()

plt.figure(figsize=(10, 6))
plt.plot(predicted_dates, data['AAPL'][-output_date:].values, linestyle='dashdot', color='green', label='Actual Price')
plt.plot(predicted_dates, add_predicted_stock_price, linestyle='dotted', color='red', label='Predicted Price')
plt.xlabel('Date', fontsize=16)
plt.ylabel('Stock Price', fontsize=16)
plt.legend(fontsize=14)
plt.title('Apple Stock Price Prediction with Trend, Seasonal, and Residual Components', fontsize=18)

# Save and log to WandB
plt.savefig('gafam_apple_stock_price_prediction_stl.png')
wandb.log({"Apple Stock Price Prediction (STL)": wandb.Image('gafam_apple_stock_price_prediction_stl.png')})

plt.show()

# End time
end_time = time.time()
runtime_seconds = end_time - start_time

# Log runtime to WandB
wandb.log({"Runtime (seconds)": runtime_seconds})

# WandB finish
wandb.finish()
