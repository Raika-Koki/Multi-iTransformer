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
    191.370,
    199.041,
    193.519,
    197.254,
    197.120
]

# マーカーと色の定義
markers = ['o', 's', '^', 'D', 'x']
colors = ['blue', 'green', 'red', 'purple', 'orange']

# グラフの作成
plt.figure(figsize=(10, 6))

for i, (category, value) in enumerate(zip(categories, values)):
    plt.scatter(i, value, color=colors[i], marker=markers[i], label=category)
    if category in ["iTransformer"]:
        plt.text(i, value - 0.8, f"{value:.2f}", ha='center', fontsize=14)
    else:
        plt.text(i, value + 0.5, f"{value:.2f}", ha='center', fontsize=14)

# 軸の設定
plt.title("Comparison of Stock Price Predictions and Real Price (AMZN 22/11/2024)", fontsize=18, pad=10)
plt.ylabel("Price ($)", fontsize=17)
plt.xlabel("Model", fontsize=17)
plt.xticks(range(len(categories)), categories, rotation=20, fontsize=14)  # フォントサイズを14に変更
plt.yticks(fontsize=14)

# 凡例の追加
plt.legend(fontsize=13) 

# グラフの表示
plt.tight_layout()
plt.savefig('chart.png')
plt.show()
