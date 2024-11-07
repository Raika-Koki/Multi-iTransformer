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
predict_stock_code = 'AAPL'
tickers = ['GOOGL', 'META', 'AMZN', 'MSFT']
start_date = '2019-12-31'
end_date = '2023-06-01'

df_stock = yf.download(predict_stock_code, start=start_date, end=end_date)
if len(df_stock) == 0:
    raise Exception("No data fetched for the given stock code and date range.")

data = df_stock['Adj Close']

# Apply STL decomposition to AAPL stock data
stl = STL(data, period=30, robust=True)
stl_series = stl.fit()

# Extract trend, seasonal, and residual components
stock_trend = stl_series.trend
stock_seasonal = stl_series.seasonal
stock_resid = stl_series.resid

# Combine STL components into a DataFrame
df_stl_components = pd.DataFrame({
    'stock_trend': stock_trend,
    'stock_seasonal': stock_seasonal,
    'stock_resid': stock_resid
})

trend_dataframes = {}
trend_dataframes[predict_stock_code] = df_stl_components[stock_trend]
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    if len(df) == 0:
        raise Exception(f"No data fetched for {ticker} for the given date range.")
    trend_dataframes[ticker] = df['Adj Close']

print(dataframes['AAPL'])

# Combine the stock data into a single DataFrame
combined_df = pd.DataFrame(dataframes)



# Normalize the STL components
df_stl_normalized, stl_mean_list, stl_std_list = data_Normalization(df_stl_components)

# Normalize the full dataset (GAFAM stock data)
df_normalized, mean_list, std_list = data_Normalization(combined_df)

# Create multivariate dataset for each component
observation_period_num = 30
predict_period_num = 1
train_rate = 0.8

# Create datasets for each component (trend, seasonal, residual)
train_data_trend, valid_data_trend = create_multivariate_dataset(
    df_normalized[['stock_trend']], observation_period_num, predict_period_num, train_rate, device)
train_data_seasonal, valid_data_seasonal = create_multivariate_dataset(
    df_normalized[['stock_seasonal']], observation_period_num, predict_period_num, train_rate, device)
train_data_resid, valid_data_resid = create_multivariate_dataset(
    df_normalized[['stock_resid']], observation_period_num, predict_period_num, train_rate, device)

# Set model parameters
epochs = 100
batch_size = 32
lr = 0.0008
step_size = 1
gamma = 0.95
depth = 2
dim = 32

# Initialize separate models for trend, seasonal, and residual components
model_trend = iTransformer(num_variates=len(tickers), lookback_len=observation_period_num, depth=depth, dim=dim, pred_length=predict_period_num).to(device)
model_seasonal = iTransformer(num_variates=len(tickers), lookback_len=observation_period_num, depth=depth, dim=dim, pred_length=predict_period_num).to(device)
model_resid = iTransformer(num_variates=len(tickers), lookback_len=observation_period_num, depth=depth, dim=dim, pred_length=predict_period_num).to(device)

# Define loss, optimizer, and scheduler
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(list(model_trend.parameters()) + list(model_seasonal.parameters()) + list(model_resid.parameters()), lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
earlystopping = EarlyStopping(patience=5)

# Log hyperparameters to WandB
wandb.config.update({
    "learning_rate": lr,
    "epochs": epochs,
    "batch_size": batch_size,
    "observation_period_num": observation_period_num,
    "predict_period_num": predict_period_num
})

# Start time
start_time = time.time()

# Training loop
for epoch in range(epochs):
    # Train trend model
    model_trend, train_loss_trend, valid_loss_trend = train(
        model_trend, train_data_trend, valid_data_trend, optimizer, criterion, scheduler, batch_size, observation_period_num)
    
    # Train seasonal model
    model_seasonal, train_loss_seasonal, valid_loss_seasonal = train(
        model_seasonal, train_data_seasonal, valid_data_seasonal, optimizer, criterion, scheduler, batch_size, observation_period_num)
    
    # Train residual model
    model_resid, train_loss_resid, valid_loss_resid = train(
        model_resid, train_data_resid, valid_data_resid, optimizer, criterion, scheduler, batch_size, observation_period_num)

    print(f"Epoch {epoch+1}/{epochs}, Trend Loss: {train_loss_trend:.4f}, Seasonal Loss: {train_loss_seasonal:.4f}, Residual Loss: {train_loss_resid:.4f}")
    
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

# Save the models
torch.save(model_trend.state_dict(), 'gafam_stock_price_trend_model.pth')
torch.save(model_seasonal.state_dict(), 'gafam_stock_price_seasonal_model.pth')
torch.save(model_resid.state_dict(), 'gafam_stock_price_resid_model.pth')

# Prediction on the validation set
model_trend.eval()
model_seasonal.eval()
model_resid.eval()

with torch.no_grad():
    # Prediction for trend, seasonal, and residual components
    last_batch_data_trend = valid_data_trend[-2][0].unsqueeze(0)
    last_batch_data_seasonal = valid_data_seasonal[-2][0].unsqueeze(0)
    last_batch_data_resid = valid_data_resid[-2][0].unsqueeze(0)

    predicted_trend = model_trend(last_batch_data_trend)
    predicted_seasonal = model_seasonal(last_batch_data_seasonal)
    predicted_resid = model_resid(last_batch_data_resid)

    

    # Select AAPL predictions
    predicted_trend_aapl = predicted_trend[0, :, 0].cpu().numpy().flatten() * stl_std_list[0] + stl_mean_list[0]
    predicted_seasonal_aapl = predicted_seasonal[0, :, 0].cpu().numpy().flatten() * stl_std_list[1] + stl_mean_list[1]
    predicted_resid_aapl = predicted_resid[0, :, 0].cpu().numpy().flatten() * stl_std_list[2] + stl_mean_list[2]

# Sum the components to get the final predicted stock price
final_predicted_aapl = predicted_trend_aapl + predicted_seasonal_aapl + predicted_resid_aapl


output_date = 10
# Plot the final result
predicted_dates_tmp = combined_df.index[-output_date:].strftime('%Y-%m-%d')
predicted_dates = predicted_dates_tmp.tolist()

plt.figure(figsize=(10, 6))
plt.plot(predicted_dates, combined_df['AAPL'][-output_date:].values, linestyle='dashdot', color='green', label='Actual Price')
plt.plot(predicted_dates, final_predicted_aapl, linestyle='dotted', color='red', label='Predicted Price')
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

# Finish WandB session
wandb.finish()
