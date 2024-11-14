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

# Download GAFAM stock data (Google, Apple, Facebook, Amazon, Microsoft)
stock_code = 'AAPL'
tickers = ['GOOGL', 'META', 'AMZN', 'MSFT']
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

for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    if len(df) == 0:
        raise Exception(f"No data fetched for {ticker} for the given date range.")
    dataframes_trend[ticker] = df['Adj Close'].squeeze()
    """print(ticker)"""
    dataframes_seasonal[ticker] = df['Adj Close'].squeeze()
    dataframes_resid[ticker] = df['Adj Close'].squeeze()

tickers = [stock_code] + tickers

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
epochs = 1000
observation_period_num = {
    "trend": 6,
    "seasonal": 48,
    "resid": 5
}
train_rates = {
    "trend": 0.9762874828359611,
    "seasonal": 0.9807915886962829,
    "resid": 0.9769493683518011
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
    earlystopping = EarlyStopping(patience=5)


    # Train model
    epochs = 1
    for epoch in range(epochs):
        model, train_loss, valid_loss = train(
            model, train_data, valid_data, optimizer, criterion, scheduler, batch_size, observation_period_num[component])
            # Early stopping based on residual validation loss
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
    #num_variates=len(tickers),
    #lookback_len=observation_period_num['trend'],
    #depth=best_params_trend['depth'],
    #dim=best_params_trend['dim'],
    #pred_length=predict_period_num,
    num_variates=len(tickers),
    lookback_len=observation_period_num['trend'],
    depth=4,
    dim=78,
    pred_length=predict_period_num
).to(device)

model_seasonal = iTransformer(
    #num_variates=len(tickers),
    #lookback_len=observation_period_num['seasonal'],
    #depth=best_params_seasonal['depth'],
    #dim=best_params_seasonal['dim'],
    #pred_length=predict_period_num
    num_variates=len(tickers),
    lookback_len=observation_period_num['seasonal'],
    depth=4,
    dim=206,
    pred_length=predict_period_num
).to(device)

model_resid = iTransformer(
    num_variates=len(tickers),
    lookback_len=observation_period_num['resid'],
    depth=3,
    dim=195,
    pred_length=predict_period_num
).to(device)

# Define loss function
criterion = nn.MSELoss()

# Define separate optimizers and schedulers for each model based on optimized parameters
"""optimizer_trend = torch.optim.AdamW(model_trend.parameters(), lr=best_params_trend['learning_rate'])
scheduler_trend = torch.optim.lr_scheduler.StepLR(optimizer_trend, step_size=best_params_trend['step_size'], gamma=best_params_trend['gamma'])

optimizer_seasonal = torch.optim.AdamW(model_seasonal.parameters(), lr=best_params_seasonal['learning_rate'])
scheduler_seasonal = torch.optim.lr_scheduler.StepLR(optimizer_seasonal, step_size=best_params_seasonal['step_size'], gamma=best_params_seasonal['gamma'])

optimizer_resid = torch.optim.AdamW(model_resid.parameters(), lr=best_params_resid['learning_rate'])
scheduler_resid = torch.optim.lr_scheduler.StepLR(optimizer_resid, step_size=best_params_resid['step_size'], gamma=best_params_resid['gamma'])
"""
optimizer_trend = torch.optim.AdamW(model_trend.parameters(), lr=0.00012140392508989015)
scheduler_trend = torch.optim.lr_scheduler.StepLR(optimizer_trend, step_size=5, gamma=0.8897669510547245)

optimizer_seasonal = torch.optim.AdamW(model_seasonal.parameters(), lr=0.0005435662922956095)
scheduler_seasonal = torch.optim.lr_scheduler.StepLR(optimizer_seasonal, step_size=5, gamma=0.8915401169867296)

optimizer_resid = torch.optim.AdamW(model_resid.parameters(), lr=0.0005701440641415547)
scheduler_resid = torch.optim.lr_scheduler.StepLR(optimizer_resid, step_size=3, gamma=0.9236096428708981)

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
    "observation_period_num": observation_period_num,
    "predict_period_num": predict_period_num
})

# Create datasets for each component (trend, seasonal, residual)
train_data_trend, valid_data_trend = create_multivariate_dataset(
    df_normalized_trend, observation_period_num["trend"], predict_period_num, train_rates["trend"], device)
train_data_seasonal, valid_data_seasonal = create_multivariate_dataset(
    df_normalized_seasonal, observation_period_num["seasonal"], predict_period_num, train_rates["seasonal"], device)
train_data_resid, valid_data_resid = create_multivariate_dataset(
    df_normalized_resid, observation_period_num["resid"], predict_period_num, train_rates["resid"], device)

# Training loop
for epoch in range(epochs):
    # Train trend model
    model_trend, train_loss_trend, valid_loss_trend = train(
        #model_trend, train_data_trend, valid_data_trend, optimizer_trend, criterion, scheduler_trend, """best_params_trend['batch_size']"""39, observation_period_num['trend'])
        model_trend, train_data_trend, valid_data_trend, optimizer_trend, criterion, scheduler_trend, 39, observation_period_num['trend'])
    
    # Train seasonal model
    model_seasonal, train_loss_seasonal, valid_loss_seasonal = train(
        #model_seasonal, train_data_seasonal, valid_data_seasonal, optimizer_seasonal, criterion, scheduler_seasonal, """best_params_seasonal['batch_size']"""139, observation_period_num['seasonal'])
        model_seasonal, train_data_seasonal, valid_data_seasonal, optimizer_seasonal, criterion, scheduler_seasonal, 139, observation_period_num['seasonal'])
    
    # Train residual model
    model_resid, train_loss_resid, valid_loss_resid = train(
        #model_resid, train_data_resid, valid_data_resid, optimizer_resid, criterion, scheduler_resid, """best_params_resid['batch_size']"""179, observation_period_num['resid'])
        model_resid, train_data_resid, valid_data_resid, optimizer_resid, criterion, scheduler_resid, 179, observation_period_num['resid'])


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
