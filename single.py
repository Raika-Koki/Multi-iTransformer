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
from statsmodels.tsa.seasonal import MSTL
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

# iTransformer関連のimport (適切にパスを修正してください)
from src.model import iTransformer, EarlyStopping
from src.data_create import data_Normalization, create_multivariate_dataset
from src.train import train

os.environ['CUDA_VISIBLE_DEVICES'] = '0' #check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 最適ハイパーパラメータをJSONファイルに保存
def save_best_hyperparameters(file_name, best_params, start_date, end_date):
    try:
        with open(file_name, 'r') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results = {}
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

# optunaの目的関数
def objective(trial, component, depth, dim):
    observation_period = trial.suggest_int('observation_period_num', 5, 252)
    train_rates = trial.suggest_float('train_rates', 0.6, 0.99)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 256)
    step_size = trial.suggest_int('step_size', 1, 15)
    gamma = trial.suggest_float('gamma', 0.75, 0.99)

    # コンポーネント別DataFrameを選択
    dataset_mapping = {
        "trend": df_normalized_trend,
        "seasonal_0": df_normalized_seasonal_0,
        "seasonal_1": df_normalized_seasonal_1,
        "seasonal_2": df_normalized_seasonal_2,
        "seasonal_3": df_normalized_seasonal_3,
        "resid": df_normalized_resid
    }
    train_data, valid_data = create_multivariate_dataset(
        dataset_mapping[component], observation_period, predict_period_num, train_rates, device
    )
    model = iTransformer(
        num_variates=1,
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
            model, train_data, valid_data, optimizer, criterion,
            scheduler, batch_size, observation_period
        )
        earlystopping(valid_loss, model)
        if earlystopping.early_stop:
            break
    return valid_loss

# MSTLでシリーズを分解する
def decompose_stock_data(data, periods, iterate, stl_kwargs):
    mstl = MSTL(data, periods=periods, iterate=iterate, stl_kwargs=stl_kwargs)
    result = mstl.fit()
    return result

# モデル生成用
def create_model(params, predict_period_num, depth, dim):
    model = iTransformer(
        num_variates=1,
        lookback_len=params['observation_period_num'],
        depth=depth,
        dim=dim,
        pred_length=predict_period_num
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=params['step_size'], gamma=params['gamma']
    )
    return model, optimizer, scheduler

# メイン処理
start_date = '2012-05-18'
initial_end_date = datetime.strptime('2024-11-25', '%Y-%m-%d')
stock_code = 'AMZN' #check
file_name = f"best_hyperparameters_{stock_code}_iTransformer(single).json"
predict_period_num = 1
depth = 4
dim = 128

while True:
    end_date = initial_end_date.strftime('%Y-%m-%d')
    df_stock = yf.download(stock_code, start=start_date, end=end_date)
    if len(df_stock) == 0:
        break

    data_Adj_close = df_stock['Adj Close'].tz_localize(None).iloc[:, 0]
    if data_Adj_close.empty:
        break

    # MSTL分解 (トレンド, 季節成分, 残差)
    periods = [252, 504, 756, 1260]
    iterate = 3
    stl_kwargs = {"seasonal_deg": 0, "inner_iter": 3, "outer_iter": 0}
    mstl_series = decompose_stock_data(data_Adj_close, periods, iterate, stl_kwargs)

    stock_trend = mstl_series.trend.tz_localize(None)
    stock_seasonal_0, stock_seasonal_1, stock_seasonal_2, stock_seasonal_3 = [
        mstl_series.seasonal.iloc[:, i].tz_localize(None) for i in range(4)
    ]
    stock_resid = mstl_series.resid.tz_localize(None)

    stock_dict = {
        "trend": stock_trend,
        "seasonal_0": stock_seasonal_0,
        "seasonal_1": stock_seasonal_1,
        "seasonal_2": stock_seasonal_2,
        "seasonal_3": stock_seasonal_3,
        "resid": stock_resid
    }

    combined_df = {}
    for comp in stock_dict:
        combined_df[comp] = pd.DataFrame({"Adj_close": stock_dict[comp].iloc[199:]})

    mean_lists, std_lists = {}, {}

    for comp in combined_df:
        df_normalized_comp, mean_list_comp, std_list_comp = data_Normalization(combined_df[comp])
        mean_lists[comp] = mean_list_comp
        std_lists[comp] = std_list_comp
        if comp == "trend":
            df_normalized_trend = df_normalized_comp
        elif comp == "seasonal_0":
            df_normalized_seasonal_0 = df_normalized_comp
        elif comp == "seasonal_1":
            df_normalized_seasonal_1 = df_normalized_comp
        elif comp == "seasonal_2":
            df_normalized_seasonal_2 = df_normalized_comp
        elif comp == "seasonal_3":
            df_normalized_seasonal_3 = df_normalized_comp
        elif comp == "resid":
            df_normalized_resid = df_normalized_comp

    wandb.init(
        project=f"{stock_code}-stock-price-prediction-by-iTransformer(single)", #check
        name=f"{stock_trend.index[0].strftime('%Y%m%d')}_{stock_trend.index[-1].strftime('%Y%m%d')}[{start_date}_{end_date}]"
    )

    best_params = {
        "trend": {}, "seasonal_0": {}, "seasonal_1": {},
        "seasonal_2": {}, "seasonal_3": {}, "resid": {}
    }
    for comp, study_name in zip(
        ["trend", "seasonal_0", "seasonal_1", "seasonal_2", "seasonal_3", "resid"],
        ["study_trend", "study_seasonal_0", "study_seasonal_1", "study_seasonal_2", "study_seasonal_3", "study_resid"]
    ):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective(t, comp, depth, dim), n_trials=50) #check
        if len(study.trials) > 0:
            best_params[comp] = study.best_params

    save_best_hyperparameters(file_name, best_params, start_date, end_date)

    predict_period_num = 1
    criterion = nn.MSELoss()
    epochs = 300 #check
    models, optimizers, schedulers = {}, {}, {}
    train_data_dict, valid_data_dict = {}, {}

    data_dict = {
        'trend': df_normalized_trend,
        'seasonal_0': df_normalized_seasonal_0,
        'seasonal_1': df_normalized_seasonal_1,
        'seasonal_2': df_normalized_seasonal_2,
        'seasonal_3': df_normalized_seasonal_3,
        'resid': df_normalized_resid
    }

    start_time = time.time()
    for comp in data_dict:
        params = best_params[comp]
        if not params:
            continue
        model, optimizer, scheduler = create_model(params, predict_period_num, depth, dim)
        models[comp] = model
        optimizers[comp] = optimizer
        schedulers[comp] = scheduler

        train_data, valid_data = create_multivariate_dataset(
            data_dict[comp], params['observation_period_num'],
            predict_period_num, params['train_rates'], device
        )
        train_data_dict[comp] = train_data
        valid_data_dict[comp] = valid_data

        earlystopping = EarlyStopping(patience=5)
        for epoch in range(epochs):
            models[comp], tr_loss, val_loss = train(
                models[comp], train_data, valid_data, optimizer, criterion,
                scheduler, params['batch_size'], params['observation_period_num']
            )
            earlystopping(val_loss, models[comp])
            if earlystopping.early_stop:
                break
            scheduler.step()
            wandb.log({f"{comp}_train_loss": tr_loss, f"{comp}_valid_loss": val_loss})

        torch.save(models[comp].state_dict(), f'{stock_code}_{comp}_model.pth')

    end_time = time.time()
    runtime_seconds = end_time - start_time
    wandb.log({"Runtime (seconds)": end_time - start_time})

    # コンポーネント予測後の合算
    if all(comp in models for comp in data_dict):
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

            predicted_trend_stock_price = predicted_trend[1][0, :, 0].cpu().numpy().flatten() * std_lists['trend'].iloc[0] + mean_lists['trend'].iloc[0]
            predicted_seasonal_0_stock_price = predicted_seasonal_0[1][0, :, 0].cpu().numpy().flatten() * std_lists['seasonal_0'].iloc[0] + mean_lists['seasonal_0'].iloc[0]
            predicted_seasonal_1_stock_price = predicted_seasonal_1[1][0, :, 0].cpu().numpy().flatten() * std_lists['seasonal_1'].iloc[0] + mean_lists['seasonal_1'].iloc[0]
            predicted_seasonal_2_stock_price = predicted_seasonal_2[1][0, :, 0].cpu().numpy().flatten() * std_lists['seasonal_2'].iloc[0] + mean_lists['seasonal_2'].iloc[0]
            predicted_seasonal_3_stock_price = predicted_seasonal_3[1][0, :, 0].cpu().numpy().flatten() * std_lists['seasonal_3'].iloc[0] + mean_lists['seasonal_3'].iloc[0]
            predicted_resid_stock_price = predicted_resid[1][0, :, 0].cpu().numpy().flatten() * std_lists['resid'].iloc[0] + mean_lists['resid'].iloc[0]

            print(predicted_trend_stock_price)
            print(predicted_seasonal_0_stock_price)
            print(predicted_seasonal_1_stock_price)
            print(predicted_seasonal_2_stock_price)
            print(predicted_seasonal_3_stock_price)
            print(predicted_resid_stock_price)

    # Sum the components to get the final predicted stock price
    final_predicted_stock_price = predicted_trend_stock_price + predicted_seasonal_0_stock_price + predicted_seasonal_1_stock_price + predicted_seasonal_2_stock_price + predicted_seasonal_3_stock_price + predicted_resid_stock_price
    
    # Prepare actual and predicted stock prices
    actual_stock_price = data_Adj_close[-predict_period_num:].values  # Actual stock price
    predicted_stock_price = final_predicted_stock_price  # Predicted stock price
    # Calculate MSE, RMSE, MAE, and R-squared
    mse = mean_squared_error(actual_stock_price, predicted_stock_price)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_stock_price, predicted_stock_price)
    r2 = r2_score(actual_stock_price, predicted_stock_price)

    # Print the results
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R-squared: {r2}")
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
        "real_stock_price": data_Adj_close.iloc[-1],
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "depth": depth,
        "dim": dim,
        "periods": periods
    })

    output_date = 10
    add_predicted_stock_price = np.append(data_Adj_close[-output_date:-predict_period_num].values, final_predicted_stock_price)
    # Plot the final result
    predicted_dates_tmp = data_Adj_close.index[-output_date:].strftime('%Y-%m-%d')
    predicted_dates = predicted_dates_tmp.tolist()
    learning_dates_tmp = data_Adj_close.index[-output_date:-1].strftime('%Y-%m-%d')
    learning_dates = learning_dates_tmp.tolist()

    plt.figure(figsize=(10, 6))
    plt.plot(predicted_dates, data_Adj_close[-output_date:].values, linestyle='dashdot', color='green', label='Actual Price')
    plt.plot(predicted_dates, add_predicted_stock_price, linestyle='dotted', color='red', label='Predicted Price')
    plt.plot(learning_dates, data_Adj_close[-output_date:-1].values, color='black', label='learning data')
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Stock Price', fontsize=16)
    plt.legend(fontsize=14)
    plt.title(f'{stock_code} Stock Price Prediction by iTransformer(single)', fontsize=18) #check

    # 保存するファイル名を一致させる
    plt.savefig(f'{stock_code.lower()}_stock_price_prediction_by_iTransformer(single).png') # 修正
    # WandBにログを送信
    wandb.log({
        f"{stock_code} Stock Price Prediction by iTransformer": wandb.Image(f'{stock_code.lower()}_stock_price_prediction_by_iTransformer(single).png') # 修正
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

