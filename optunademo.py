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

# WandB initialization
wandb.init(project="gafam-stock-price-prediction")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download GAFAM stock data (Google, Apple, Facebook, Amazon, Microsoft)
stock_code = 'AAPL'
tickers = ['GOOGL', 'META', 'AMZN', 'MSFT']
start_date = '2012-05-18'
end_date = '2023-06-01'

df_stock = yf.download(stock_code, start=start_date, end=end_date)
if len(df_stock) == 0:
    raise Exception("No data fetched for the given stock code and date range.")

data = df_stock['Adj Close']

# Apply STL decomposition to AAPL stock data
stl = STL(data, period=252, robust=True)
stl_series = stl.fit()

# Extract trend, seasonal, and residual components
stock_trend = stl_series.trend
stock_seasonal = stl_series.seasonal
stock_resid = stl_series.resid

dataframes_trend = {stock_code: stock_trend}
dataframes_seasonal = {stock_code: stock_seasonal}
dataframes_resid = {stock_code: stock_resid}

for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    if len(df) == 0:
        raise Exception(f"No data fetched for {ticker} for the given date range.")
    dataframes_trend[ticker] = df['Adj Close']
    dataframes_seasonal[ticker] = df['Adj Close']
    dataframes_resid[ticker] = df['Adj Close']

tickers = [stock_code] + tickers

# Combine the stock data into a single DataFrame
combined_df_trend = pd.DataFrame(dataframes_trend)
combined_df_seasonal = pd.DataFrame(dataframes_seasonal)
combined_df_resid = pd.DataFrame(dataframes_resid)

# Normalize the full dataset (GAFAM stock data)
df_normalized_trend, mean_list_trend, std_list_trend = data_Normalization(combined_df_trend)
df_normalized_seasonal, mean_list_seasonal, std_list_seasonal = data_Normalization(combined_df_seasonal)
df_normalized_resid, mean_list_resid, std_list_resid = data_Normalization(combined_df_resid)

# Create multivariate dataset for each component
predict_period_num = 1
observation_period_num = {
    "trend": 30,
    "seasonal": 30,
    "resid": 30
}
train_rates = {
    "trend": 0.85,
    "seasonal": 0.85,
    "resid": 0.85
}

# Optuna objective function for hyperparameter tuning
def objective(trial, component):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    batch_size = trial.suggest_int('batch_size', 16, 256)
    step_size = trial.suggest_int('step_size', 1, 15)
    gamma = trial.suggest_float('gamma', 0.75, 0.99)
    depth = trial.suggest_int('depth', 2, 6)
    dim = trial.suggest_int('dim', 16, 256)

    # Define the dataset and model for the component
    if component == "trend":
        train_data, valid_data = create_multivariate_dataset(
            df_normalized_trend, observation_period_num["trend"], predict_period_num, train_rates["trend"], device)
        model = iTransformer(num_variates=len(tickers), lookback_len=observation_period_num["trend"], depth=depth, dim=dim, pred_length=predict_period_num).to(device)
    elif component == "seasonal":
        train_data, valid_data = create_multivariate_dataset(
            df_normalized_seasonal, observation_period_num["seasonal"], predict_period_num, train_rates["seasonal"], device)
        model = iTransformer(num_variates=len(tickers), lookback_len=observation_period_num["seasonal"], depth=depth, dim=dim, pred_length=predict_period_num).to(device)
    else:
        train_data, valid_data = create_multivariate_dataset(
            df_normalized_resid, observation_period_num["resid"], predict_period_num, train_rates["resid"], device)
        model = iTransformer(num_variates=len(tickers), lookback_len=observation_period_num["resid"], depth=depth, dim=dim, pred_length=predict_period_num).to(device)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.MSELoss()

    # Train model
    model, train_loss, valid_loss = train(
        model, train_data, valid_data, optimizer, criterion, scheduler, batch_size, observation_period_num[component])

    return valid_loss

# Optimize for each component
study_trend = optuna.create_study(direction='minimize')
study_trend.optimize(lambda trial: objective(trial, "trend"), n_trials=50)

study_seasonal = optuna.create_study(direction='minimize')
study_seasonal.optimize(lambda trial: objective(trial, "seasonal"), n_trials=50)

study_resid = optuna.create_study(direction='minimize')
study_resid.optimize(lambda trial: objective(trial, "resid"), n_trials=50)

# Log best hyperparameters
print("Best hyperparameters (trend):", study_trend.best_params)
print("Best hyperparameters (seasonal):", study_seasonal.best_params)
print("Best hyperparameters (resid):", study_resid.best_params)

# WandB finish
wandb.finish()
