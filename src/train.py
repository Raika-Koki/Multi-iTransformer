import torch
from src.data_create import get_batch

def train(model, train_data, valid_data, optimizer, criterion, scheduler, batch_size, observation_period_num):
    """
    モデルのトレーニングを行う関数
    model: トレーニングするモデル
    train_data: トレーニングデータ
    valid_data: 検証データ
    optimizer: 最適化関数
    criterion: 損失関数
    scheduler: 学習率スケジューラ
    batch_size: バッチサイズ
    observation_period_num: 観測期間の長さ
    """

    # トレーニングモードに設定
    model.train()  

    total_loss_train = 0.0

    for batch, i in enumerate(range(0, len(train_data), batch_size)):

        # バッチデータを取得
        data, targets = get_batch(train_data, i, batch_size, observation_period_num)

        # 勾配の初期化
        optimizer.zero_grad()

        # モデルにデータを入力
        output = model(data)

        # 出力の形式を調整
        if isinstance(output, dict):
            output = output[list(output.keys())[0]]  # 最初のキーの値を取得

        # 損失計算
        loss = criterion(output, targets)  
        loss.backward()  # 逆伝播
        optimizer.step()  # 重みの更新

        total_loss_train += loss.item() * data.size(0)

    # 学習率スケジューラをステップ
    scheduler.step()

    # トレーニングデータ全体での平均損失を計算
    total_loss_train = total_loss_train / len(train_data)

    # 検証モード
    model.eval()
    total_loss_valid = 0.0

    with torch.no_grad():
        for i in range(0, len(valid_data), batch_size):
            # 検証データのバッチを取得
            data, targets = get_batch(valid_data, i, batch_size, observation_period_num)

            # モデルにデータを入力
            output = model(data)

            if isinstance(output, dict):
                output = output[list(output.keys())[0]]  # 最初のキーの値を取得

            # 損失計算
            loss = criterion(output, targets)
            total_loss_valid += loss.item() * data.size(0)

    # 検証データ全体での平均損失を計算
    total_loss_valid = total_loss_valid / len(valid_data)

    return model, total_loss_train, total_loss_valid
