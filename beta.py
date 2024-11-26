import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta

# パラメータ設定
alpha = 8
beta_param = 3
weight = 0.5
start_date = '2023-01-01'
end_date = '2023-06-01'

# 全ての日付範囲を生成
date_range = pd.date_range(start=start_date, end=end_date)

# CPIの発表日（例として与えられたリストを使用）
cpi_release_dates = pd.to_datetime([
    '2023-01-15', '2023-02-14', '2023-03-13', '2023-04-15', 
    '2023-05-11'
])

# 関心度を保持する空のデータフレームを作成
interest_levels = pd.DataFrame({"Date": date_range, "Interest_Level": 0.0})

# 各CPI発表日の関心度を計算
for release_date in cpi_release_dates:
    # 発表日の前後5日間の範囲
    day_offsets = (interest_levels["Date"] - release_date).dt.days
    within_range = day_offsets.between(-3, 7)
    
    # ベータ分布のスケールとシフトを適用
    x_values = (day_offsets[within_range] + 5) / 10  # 発表日 (-5, +5) を (0, 1) に変換

    # ベータ分布に基づいて関心度を計算
    beta_values = beta.pdf(x_values, alpha, beta_param)
    beta_values *= weight / beta_values.max()  # 重みに基づき正規化

    # 発表日周辺の日付に関心度を割り当て
    interest_levels.loc[within_range, "Interest_Level"] += beta_values

# 結果を表示
non_zero_interest = interest_levels[interest_levels["Interest_Level"] > 0]
print(non_zero_interest.head(20))  # 関心度が非ゼロの日付のみ表示

# プロット
plt.figure(figsize=(15, 6))
plt.plot(interest_levels["Date"], interest_levels["Interest_Level"], label="Interest Level")
plt.scatter(cpi_release_dates, [weight] * len(cpi_release_dates), color='red', label="CPI Release Dates", zorder=5)
plt.title("Trader Interest Level Around CPI Releases")
plt.xlabel("Date")
plt.ylabel("Interest Level")
plt.legend()
plt.grid()
plt.show()
