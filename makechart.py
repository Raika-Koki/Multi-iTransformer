import textwrap
import matplotlib.pyplot as plt

# データの定義
categories = [
    "Real Stock Price",
    "Multi iTransformer with MSTL (Proposed)",
    "Multi Transformer with MSTL",
    "Multi iTransformer",
    "Multi Transformer",
    "Single iTransformer with MSTL"
]
values = [
    197.120,
    197.254,
    193.519,
    199.041,
    191.370,
    188.666
]


markers = ['^', 'D', 'x', 'o', 's', '*']
colors = ['red', 'purple', 'orange', 'green', 'blue', 'brown']

plt.figure(figsize=(10, 6))

for i, (category, value) in enumerate(zip(categories, values)):
    plt.scatter(i, value, color=colors[i], marker=markers[i], label=category)
    if category in ["Multi iTransformer with MSTL (Proposed)", "Multi iTransformer", "Real Stock Price"]:
        plt.text(i, value - 1.0, f"{value:.2f}", ha='center', fontsize=14)
    else:
        plt.text(i, value + 0.6, f"{value:.2f}", ha='center', fontsize=14)

wrapped_labels = ['\n'.join(textwrap.wrap(cat, 15)) for cat in categories]
plt.title("Comparison of Stock Price Predictions and Real Price (AMZN 22/11/2024)", fontsize=18, pad=10)
plt.ylabel("Price ($)", fontsize=17)
plt.xlabel("Model", fontsize=17)
ticks = plt.xticks(range(len(categories)), wrapped_labels, fontsize=14)

# 赤文字にする
for i, label in enumerate(ticks[1]):
    if categories[i] == "Multi iTransformer with MSTL (Proposed)":
        label.set_color('red')

plt.yticks(fontsize=14)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig('chart.png')
plt.show()
