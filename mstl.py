"""import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import MSTL

# 株価データのダウンロード
stock_code = 'AAPL'
start_date = '2012-05-18'
end_date = '2023-06-01'

df_stock = yf.download(stock_code, start=start_date, end=end_date)
if len(df_stock) == 0:
    raise Exception("No data fetched for the given stock code and date range.")

data = df_stock['Adj Close']
print(data)

# MSTLで株価データを分解する
# 252日、504日、756日、1260日の季節性周期を設定
mstl = MSTL(data, periods=[252, 504, 756, 1260], iterate=3, 
        stl_kwargs={"seasonal_deg": 0, "inner_iter": 3, "outer_iter": 1})
result = mstl.fit()


# 分解結果のプロット
fig, axes = plt.subplots(7, 1, figsize=(12, 12), sharex=True)

# 元のデータ
axes[0].plot(data, label='Original', color='black')
axes[0].legend(loc='upper left')
axes[0].set_title('Original Series')

# トレンド成分
axes[1].plot(result.trend, label='Trend', color='blue')
axes[1].legend(loc='upper left')
axes[1].set_title('Trend Component')

# 126日季節性
axes[2].plot(result.seasonal[:, 0], label='Seasonal (252 days)', color='green')
axes[2].legend(loc='upper left')
axes[2].set_title('Seasonal (252 days)')

# 252日季節性
axes[3].plot(result.seasonal[:, 1], label='Seasonal (504 days)', color='orange')
axes[3].legend(loc='upper left')
axes[3].set_title('Seasonal (504 days)')

# 1260日季節性
axes[4].plot(result.seasonal[:, 2], label='Seasonal (756 days)', color='red')
axes[4].legend(loc='upper left')
axes[4].set_title('Seasonal (756 days)')

# 2520日季節性
axes[5].plot(result.seasonal[:, 3], label='Seasonal (1260 days)', color='purple')
axes[5].legend(loc='upper left')
axes[5].set_title('Seasonal (1260 days)')

# 2520日季節性
axes[6].plot(result.resid, label='resid', color='grey')
axes[6].legend(loc='upper left')
axes[6].set_title('resid')

plt.tight_layout()
plt.show()
plt.savefig("output.png")

# 残差の標準偏差
resid_std = result.resid.std()

# 各季節性成分の標準偏差を計算して、残差の標準偏差と比較
for i, period in enumerate([252, 504, 756, 1260]):
    seasonal_std = result.seasonal[:, i].std()  # 季節性成分の標準偏差
    seasonal_strength = seasonal_std / resid_std  # 季節性の強さ
    print(f"Seasonal ({period} days) の季節性の強さ: {seasonal_strength:.2f}")

"""

import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import MSTL
import pandas as pd

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

# 株価データのダウンロード
stock_code = 'AAPL'
start_date = '2012-05-18'
end_date = '2023-06-01'

df_stock = yf.download(stock_code, start=start_date, end=end_date)
if len(df_stock) == 0:
    raise Exception("No data fetched for the given stock code and date range.")

data = df_stock['Adj Close']

# MSTLによる分解
periods = [252, 504, 756, 1260]
iterate = 3
stl_kwargs = {"seasonal_deg": 0, "inner_iter": 3, "outer_iter": 1}

result = decompose_stock_data(data, periods, iterate, stl_kwargs)

# 分解結果のプロット
print(result.trend)
# 季節性成分の1つ目の周期の表示
print(f"Seasonal Component ({periods[0]} days):")
print(result.seasonal.iloc[:, 3])  # .ilocで列を指定


# 分解結果のプロット
fig, axes = plt.subplots(7, 1, figsize=(12, 12), sharex=True)

# 元のデータ
axes[0].plot(data, label='Original', color='black')
axes[0].legend(loc='upper left')
axes[0].set_title('Original Series')

# トレンド成分
axes[1].plot(result.trend, label='Trend', color='blue')
axes[1].legend(loc='upper left')
axes[1].set_title('Trend Component')

# 季節性成分のプロット
colors = ['green', 'orange', 'red', 'purple']
labels = ['Seasonal (252 days)', 'Seasonal (504 days)', 'Seasonal (756 days)', 'Seasonal (1260 days)']
for i in range(4):
    axes[i + 2].plot(result.seasonal.iloc[:, i], label=labels[i], color=colors[i])
    axes[i + 2].legend(loc='upper left')
    axes[i + 2].set_title(labels[i])

# 残差成分
axes[6].plot(result.resid, label='Resid', color='grey')
axes[6].legend(loc='upper left')
axes[6].set_title('Residual Component')

plt.tight_layout()
plt.show()
plt.savefig("output.png")
