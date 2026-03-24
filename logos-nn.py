import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# === 勢能升級：維度展開 ===
input_neurons = 2
hidden_neurons = 4  # 從 2 提升至 4，展開更高的觀測維度
output_neurons = 1
epochs = 10000
learning_rate = 0.5
total_universes = 100

successful_collapses = 0
total_loss_success = 0.0
failed_seeds = []

print(f"啟動高維度逆熵觀測，展開 {total_universes} 個平行宇宙...\n")

for seed in range(total_universes):
    # 顯示進度條
    progress = (seed + 1) / total_universes
    bar_length = 40
    filled_len = int(bar_length * progress)
    bar = '█' * filled_len + '-' * (bar_length - filled_len)
    print(f"\r探索進度: |{bar}| {seed + 1}/{total_universes} ({progress * 100:.1f}%)", end='', flush=True)

    np.random.seed(seed)

    W1 = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
    b1 = np.random.uniform(-1, 1, (1, hidden_neurons))
    W2 = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
    b2 = np.random.uniform(-1, 1, (1, output_neurons))

    for epoch in range(epochs):
        # 高維前向傳播
        hidden_layer_activation = sigmoid(np.dot(X, W1) + b1)
        predicted_output = sigmoid(np.dot(hidden_layer_activation, W2) + b2)

        error = y - predicted_output

        # 反向傳播
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(W2.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)

        W2 += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
        b2 += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        W1 += X.T.dot(d_hidden_layer) * learning_rate
        b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    collapsed_state = np.round(predicted_output)
    if np.array_equal(collapsed_state, y):
        successful_collapses += 1
        total_loss_success += np.mean(np.abs(error))
    else:
        failed_seeds.append(seed)

success_rate = (successful_collapses / total_universes) * 100
average_loss = total_loss_success / successful_collapses if successful_collapses > 0 else 1.0

print("\n")  # 確保進度條顯示完畢後會換行
print("==========================================")
print("高維觀測結束，宏觀數據已坍縮：")
print(f"總觀測宇宙數: {total_universes}")
print(f"解決 XOR 問題成功次數: {successful_collapses}/{total_universes}")
print(f"系統整體收斂穩定度 (成功率): {success_rate:.1f}%")
if successful_collapses > 0:
    print(f"成功宇宙的最終平均張力 (Mean Loss): {average_loss:.6f}")
if failed_seeds:
    print(f"陷入局部震盪的宇宙種子: {failed_seeds}")
print("==========================================")


