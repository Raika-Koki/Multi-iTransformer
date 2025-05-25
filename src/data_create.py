import torch
import numpy as np

"""def data_Normalization(df):
    """"""
    データの正規化を行う関数
    df: データフレーム (株価など)
    """"""
    mean_list = df.mean()
    std_list = df.std()
    df = (df - mean_list) / std_list
    return df, mean_list, std_list"""
def data_Normalization(df):
    mean_list = df.mean()
    std_list = df.std()
    std_list[std_list == 0] = 1  # 標準偏差がゼロの場合は1に置き換える
    df = (df - mean_list) / std_list
    return df, mean_list, std_list

def create_multivariate_dataset(data_norm, observation_period_num, predict_period_num, train_rate, device):
    """
    多変量時系列データセットを作成する関数
    data_norm: 正規化済みのデータフレーム
    observation_period_num: 観測期間の長さ
    predict_period_num: 予測期間の長さ
    train_rate: トレーニングデータの割合
    device: 使用するデバイス (CPU/GPU)
    """
    
    inout_data = []
    # 全銘柄のデータを使って時系列データセットを作成
    for i in range(len(data_norm) - observation_period_num - predict_period_num):
        # 観測期間中のデータを抽出 (多変量データ)
        data = data_norm.iloc[i:i + observation_period_num].values  # [観測期間, 銘柄数]

        # 予測期間中のデータを抽出（複数銘柄の次の時間のデータ）
        label = data_norm.iloc[i + observation_period_num:i + observation_period_num + predict_period_num].values

        # 形状を揃えるためにリストに追加
        inout_data.append((data, label))


    # データをテンソルに変換し、デバイスに移動
    inout_data = [(torch.tensor(data, dtype=torch.float32).to(device), 
                   torch.tensor(label, dtype=torch.float32).to(device)) for data, label in inout_data]

    # トレーニングとバリデーションデータに分割
    train_data = inout_data[:int(len(inout_data) * train_rate)]
    valid_data = inout_data[int(len(inout_data) * train_rate):]

    return train_data, valid_data

def get_batch(source, i, batch_size, observation_period_num):
    """
    バッチを取得する関数
    source: データセット (リスト形式)
    i: バッチのスタートインデックス
    batch_size: バッチサイズ
    observation_period_num: 観測期間の長さ
    """
    data = source[i:i+batch_size]
    inputs = torch.stack([item[0] for item in data])  # [バッチサイズ, 観測期間, 銘柄数]
    targets = torch.stack([item[1] for item in data])  # [バッチサイズ, 予測期間, 銘柄数]

    return inputs, targets
