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

# WandB initialization
wandb.init(project="gafam-stock-price-prediction")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download GAFAM stock data (Google, Apple, Facebook, Amazon, Microsoft)
tickers = ['AAPL', 'GOOGL', 'META', 'AMZN', 'MSFT']
start_date = '2019-12-31'
end_date = '2023-06-01'

dataframes = {}
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    if len(df) == 0:
        raise Exception(f"No data fetched for {ticker} for the given date range.")
    dataframes[ticker] = df['Adj Close']

print(dataframes['AAPL'])

# Combine the stock data into a single DataFrame
combined_df = pd.DataFrame(dataframes)

# Normalize the data
df_normalized, mean_list, std_list = data_Normalization(combined_df)

# Create multivariate dataset
observation_period_num = 30
predict_period_num = 1
train_rate = 0.8
train_data, valid_data = create_multivariate_dataset(df_normalized, observation_period_num, predict_period_num, train_rate, device)
print("Dataset created successfully.")

# Set model parameters
epochs = 100
batch_size = 32
lr = 0.0008
step_size = 1
gamma = 0.95
depth = 2
dim = 32

# Initialize the Transformer model for multivariate forecasting
model = iTransformer(
    num_variates=len(tickers),  # GAFAM: 5 stocks
    lookback_len=observation_period_num,
    depth=depth,
    dim=dim,
    pred_length=predict_period_num
).to(device)

# Define loss, optimizer, and scheduler
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr)
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
    model, train_loss, valid_loss = train(model, train_data, valid_data, optimizer, criterion, scheduler, batch_size, observation_period_num)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")

    # Log loss to WandB
    wandb.log({"Train Loss": train_loss, "Validation Loss": valid_loss})

    earlystopping(valid_loss, model)
    if earlystopping.early_stop:
        print("Early stopping")
        break

# Save the model
torch.save(model.state_dict(), 'gafam_stock_price_transformer_model.pth')

output_date = 10

# Prediction on the validation set
model.eval()
with torch.no_grad():
    last_batch_data, last_batch_target = valid_data[-2]
    last_batch_data = last_batch_data.unsqueeze(0)
    last_batch_data = last_batch_data.view(1, model.lookback_len, model.num_variates)
    
    # Get the prediction from the model
    predicted_stock_price = model(last_batch_data)
    
    if isinstance(predicted_stock_price, dict):
        predicted_stock_price = predicted_stock_price[list(predicted_stock_price.keys())[0]]

    # Since you're predicting multiple stocks, we extract only the prediction for Apple (AAPL).
    predicted_selected_stock_price = predicted_stock_price[0, :, 0]  # Assuming AAPL is the first stock in the dataset

# Denormalize predicted Apple stock prices
predicted_selected_stock_price = predicted_selected_stock_price.cpu().numpy().flatten() * std_list[0] + mean_list[0]  # Using AAPL normalization factors
actual_stock_price = combined_df['AAPL'][-output_date:].values
training_stock_price = combined_df['AAPL'][-output_date:-1].values
add_predicted_stock_price = np.append(combined_df['AAPL'][-output_date:-predict_period_num].values, predicted_selected_stock_price)

# Generate date lists for plotting
predicted_dates_tmp = combined_df.index[-output_date:].strftime('%Y-%m-%d')
predicted_dates = predicted_dates_tmp.tolist()
training_dates = predicted_dates_tmp[:-1].tolist()


print(predicted_dates)
print(predicted_selected_stock_price)
print(add_predicted_stock_price)
print(actual_stock_price)


# Plot results
plt.figure(figsize=(10, 6))
plt.plot(predicted_dates, actual_stock_price, linestyle='dashdot', color='green', label='Actual Price')
plt.plot(predicted_dates, add_predicted_stock_price, linestyle='dotted', color='red', label='Predicted Price')
plt.plot(training_dates, training_stock_price, color='black', label='Training Data')
plt.xlabel('Date', fontsize=16)
plt.ylabel('Stock Price', fontsize=16)
plt.legend(fontsize=14)
plt.title('Apple Stock Price Prediction using GAFAM Data', fontsize=18)

# Save and log to WandB
plt.savefig('gafam_apple_stock_price_prediction.png')
wandb.log({"Apple Stock Price Prediction": wandb.Image('gafam_apple_stock_price_prediction.png')})

# Show plot
plt.show()

# End time
end_time = time.time()
runtime_seconds = end_time - start_time

# Log runtime to WandB
wandb.log({"Runtime (seconds)": runtime_seconds})

# Finish WandB session
wandb.finish()
