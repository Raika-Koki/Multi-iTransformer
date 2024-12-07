"""import os
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd
import requests
from datetime import datetime

# APIキーを.envファイルから読み込む
load_dotenv("api_key.env")

# 各APIのキーを取得
FRED_API = os.getenv("FRED_API")
BEA_API = os.getenv("BEA_API")
BLS_API = os.getenv("BLS_API")

# 期間設定
start_date = "2012-05-18"
end_date = "2023-06-01"

# 結果を保存する辞書
data_results = {}

# --- FRED API ---
def get_fred_data(series_id, start_date, end_date, api_key):
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    return data

# PCE, Monetary Base, FF金利などのデータを取得
fred_series = {
    "PCE": "PCE",  # Personal Consumption Expenditures
    "Monetary Base": "AMBNS",  # Monetary Base
    "FFR": "FEDFUNDS",  # Federal Funds Rate
}
for key, series_id in fred_series.items():
    data_results[key] = get_fred_data(series_id, start_date, end_date, FRED_API)

# --- BEA API ---
def get_bea_data(dataset_name, table_name, start_date, end_date, api_key):
    url = "https://apps.bea.gov/api/data/"
    params = {
        "UserID": api_key,
        "method": "GetData",
        "datasetname": dataset_name,
        "TableName": table_name,
        "Frequency": "Q",  # 四半期データ
        "Year": f"{start_date[:4]},{end_date[:4]}",
        "ResultFormat": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    # データ整形
    results = []
    for item in data["BEAAPI"]["Results"]["Data"]:
        date = f"{item['TimePeriod']}-01"  # 年月を日付形式に
        value = item["DataValue"]
        results.append({"Date": date, "Value": value})
    return pd.DataFrame(results)

# GDPのデータ取得
data_results["GDP"] = get_bea_data("NIPA", "T10101", start_date, end_date, BEA_API)

# --- BLS API ---
def get_bls_data(series_id, start_date, end_date, api_key):
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {"Content-Type": "application/json"}
    payload = {
        "seriesid": [series_id],
        "startyear": start_date[:4],
        "endyear": end_date[:4],
        "registrationkey": api_key
    }
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()
    # データ整形
    results = []
    for entry in data["Results"]["series"][0]["data"]:
        date = f"{entry['year']}-{entry['period'][1:]}-01"  # 年+月
        value = entry["value"]
        results.append({"Date": date, "Value": value})
    return pd.DataFrame(results)

# PPIとEmployment Situationのデータ取得
bls_series = {
    "PPI": "WPU00000000",  # Producer Price Index
    "Employment Situation": "CES0000000001",  # All employees, total nonfarm
}
for key, series_id in bls_series.items():
    data_results[key] = get_bls_data(series_id, start_date, end_date, BLS_API)

# --- 結果を保存 ---
for key, df in data_results.items():
    print(f"--- {key} Data ---")
    print(df.head())
    # CSVファイルとして保存（必要に応じてコメントを外してください）
    # df.to_csv(f"{key}_data.csv", index=False)

"""


import os
from dotenv import load_dotenv
import pandas as pd
from fredapi import Fred
import requests

# .envファイルからAPIキーを読み込む
load_dotenv("api_key.env")
fred_api_key = os.getenv("FRED_API")


# FREDクライアントの設定
fred = Fred(api_key=fred_api_key)

# 期間設定
start_date = '2012-05-18'
end_date = '2023-06-01'

# 経済指標の取得関数
def fetch_fred_data(series_id, indicator_name):
    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    return pd.DataFrame({indicator_name: data}).reset_index().rename(columns={"index": "Date"})

# 1. CPI (Consumer Price Index)
cpi_data = fetch_fred_data('CPIAUCSL', 'CPI')

# 2. PCE (Personal Consumption Expenditures)
pce_data = fetch_fred_data('PCE', 'PCE')

# 3. PPI (Producer Price Index)
ppi_data = fetch_fred_data('PPIACO', 'PPI')

# 4. Unemployment Rate (BLS)
unemployment_data = fetch_fred_data('UNRATE', 'Unemployment Rate')

# 5. Nonfarm Payrolls (BLS)
nonfarm_data = fetch_fred_data('PAYEMS', 'Nonfarm Payrolls')

# 6. GDP (Gross Domestic Product)
gdp_data = fetch_fred_data('GDP', 'GDP')

# 7. FF Rate (Federal Funds Rate)
ff_rate_data = fetch_fred_data('FEDFUNDS', 'Federal Funds Rate')

# 8. Monetary Base
monetary_base_data = fetch_fred_data('AMBNS', 'Monetary Base')

# 各データを表示
print("\nCPI Data:\n", cpi_data)
print("\nPCE Data:\n", pce_data)
print("\nPPI Data:\n", ppi_data)
print("\nUnemployment Rate Data:\n", unemployment_data)
print("\nNonfarm Payroll Data:\n", nonfarm_data)
print("\nGDP Data:\n", gdp_data)
print("\nFederal Funds Rate Data:\n", ff_rate_data)
print("\nMonetary Base Data:\n", monetary_base_data)




