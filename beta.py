"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
import json

#日付範囲を設定
start_date = '2023-01-01'
end_date = '2023-06-01'


# JSONファイルからデータを読み込む
with open('Release_Dates.json', 'r') as file:
    Release_Dates = json.load(file)
    
# 全ての日付範囲を生成
date_range = pd.date_range(start=start_date, end=end_date)

# パラメータ設定
alpha = 8
beta_param = 3
weight = 0.5

# 指定された期間のCPI発表日を抽出
release_dates = []
found_post_end_date = False
pre_start_date_release_date = None
for year in Release_Dates['CPI']:
    for month in Release_Dates['CPI'][year]:
        release_date = Release_Dates['CPI'][year][month]
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


# 関心度を保持する空のデータフレームを作成
interest_levels = pd.DataFrame({"Date": date_range, "Interest_Level": 0.0})
beta_values_max = 0

# 各CPI発表日の関心度を計算
for release_date in release_dates:
    # 発表日の前後5日間の範囲
    day_offsets = (interest_levels["Date"] - release_date).dt.days
    within_range = day_offsets.between(-7, 2)
    #print(day_offsets)
    
    # ベータ分布のスケールとシフトを適用
    x_values = (day_offsets[within_range] + 8) / 10  # 発表日 (-5, +5) を (0, 1) に変換
    #print(x_values)

    if len(x_values) > 0:
        # ベータ分布に基づいて関心度を計算
        beta_values = beta.pdf(x_values, alpha, beta_param)
        print(beta_values)
        if beta_values_max == 0:
            beta_values_max = beta_values.max()

        beta_values *= weight / beta_values_max  # 重みに基づき正規化
        

        # 発表日周辺の日付に関心度を割り当て
        interest_levels.loc[within_range, "Interest_Level"] += beta_values

print(interest_levels)

# 結果を表示
non_zero_interest = interest_levels[interest_levels["Interest_Level"] > 0]
print(non_zero_interest.head(20))  # 関心度が非ゼロの日付のみ表示
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
import json

# JSONファイルからデータを読み込む
with open('Release_Dates.json', 'r') as file:
    release_dates_data = json.load(file)

# パラメータ設定
alpha = 8
beta_param = 3
weight = 0.5
start_date = '2023-01-01'
end_date = '2023-06-01'

# 全ての日付範囲を生成
date_range = pd.date_range(start=start_date, end=end_date)

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

# 指標ごとに関心度を計算して出力
for indicator in ['CPI', 'PCE', 'PPI', 'Unemployment_Rate', 'Nonfarm_Payrolls_Release_Dates', 'Monetary_Base_Data', 'Federal_Funds_Rate_Release_Dates']:
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
    print(interest_levels)


    # 結果を表示
    non_zero_interest = interest_levels[interest_levels["Interest_Level"] > 0]
    #print(f"Interest levels for {indicator}:")
    #print(non_zero_interest.head(20))  # 関心度が非ゼロの日付のみ表示

