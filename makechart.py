import matplotlib.pyplot as plt

# データの定義
categories = [
    "Transformer",
    "iTransformer",
    "Transformer with MSTL",
    "iTransformer with MSTL",
    "Real Stock Price"
]
values = [
    222.0615234375,
    225.5749969482422,
    228.17929077148438,
    229.91355895996094,
    229.8699951171875
]

# グラフの作成
plt.figure(figsize=(10, 6))
plt.bar(categories, values, color='skyblue')

# グラフの装飾
plt.title("Comparison of Stock Price Predictions and Real Price", fontsize=14)
plt.ylabel("Price", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=20, fontsize=10)
plt.yticks(fontsize=10)

# 値をバーの上に表示
for i, value in enumerate(values):
    plt.text(i, value + 0.5, f"{value:.2f}", ha='center', fontsize=10)

# グラフの表示
plt.tight_layout()
plt.savefig('chart.png')
plt.show()
