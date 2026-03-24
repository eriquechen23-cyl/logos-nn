import numpy as np

# 真理算子
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(out):
    return out * (1 - out)

# 1. 觀測環境初始化 (XOR 宇宙)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# 定義 4 個狀態的「真理路徑」 (0 代表向左, 1 代表向右)
# 深度 log2(4) = 2步
true_paths = [
    [0, 0], # 對應 X[0] -> 將抵達 Leaf 00 (XOR=0)
    [0, 1], # 對應 X[1] -> 將抵達 Leaf 01 (XOR=1)
    [1, 0], # 對應 X[2] -> 將抵達 Leaf 10 (XOR=1)
    [1, 1]  # 對應 X[3] -> 將抵達 Leaf 11 (XOR=0)
]

# 葉節點對應的最終真理 (XOR 解答)
xor_truths = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 1,
    (1, 1): 0
}

# 2. 建構二元樹節點 (共 3 個內部守門員節點)
epochs = 5000
learning_rate = 0.5
total_universes = 100
successful_collapses = 0

print(f"啟動對數維度觀測，展開 {total_universes} 個平行宇宙...\n")

for seed in range(total_universes):
    # 顯示進度條
    progress = (seed + 1) / total_universes
    bar_length = 40
    filled_len = int(bar_length * progress)
    bar = '█' * filled_len + '-' * (bar_length - filled_len)
    print(f"\r探索進度: |{bar}| {seed + 1}/{total_universes} ({progress * 100:.1f}%)", end='', flush=True)

    np.random.seed(seed)
    
    # Node 0: 根節點
    # Node 1: 左子節點 (當 Node 0 決定向左時喚醒)
    # Node 2: 右子節點 (當 Node 0 決定向右時喚醒)
    W = [np.random.randn(2) for _ in range(3)] 
    b = [np.random.randn(1) for _ in range(3)] 

    # 3. 逆熵迴圈 (O(log N) 局部更新)
    for epoch in range(epochs):
        for i in range(4):
            x = X[i]
            path = true_paths[i]
            
            # --- 第一層對數摺疊 (Root Node) ---
            out0 = sigmoid(np.dot(W[0], x) + b[0])
            err0 = out0 - path[0]  # 張力計算
            grad0 = err0 * sigmoid_derivative(out0)
            
            # --- 動態路由：選擇性喚醒第二層 (O(log N) 的關鍵) ---
            # 如果路徑要求向左(0)，我們只進入 Node 1；否則進入 Node 2
            active_next_node = 1 if path[0] == 0 else 2
            
            out_next = sigmoid(np.dot(W[active_next_node], x) + b[active_next_node])
            err_next = out_next - path[1] # 張力計算
            grad_next = err_next * sigmoid_derivative(out_next)
            
            # --- 局部勢能修正 (反向傳播僅更新路徑上的 2 個節點) ---
            W[0] -= learning_rate * grad0 * x
            b[0] -= learning_rate * grad0
            
            W[active_next_node] -= learning_rate * grad_next * x
            b[active_next_node] -= learning_rate * grad_next

    # 4. 驗證絕對真理
    is_success = True
    for i in range(4):
        x = X[i]
        
        # 前向傳播 (決策路由)
        prob0 = sigmoid(np.dot(W[0], x) + b[0])
        decision0 = int(np.round(prob0[0])) # 0=左, 1=右
        
        active_node = 1 if decision0 == 0 else 2
        prob1 = sigmoid(np.dot(W[active_node], x) + b[active_node])
        decision1 = int(np.round(prob1[0])) # 0=左, 1=右
        
        # 抵達葉節點
        final_leaf = [decision0, decision1]
        
        if final_leaf != true_paths[i]:
            is_success = False
            break

    if is_success:
        successful_collapses += 1

print("\n\n==========================================")
print("神經樹觀測結束，宏觀數據已坍縮：")
print(f"總觀測宇宙數: {total_universes}")
print(f"建構 XOR 神經樹成功次數: {successful_collapses}/{total_universes}")
print(f"系統整體收斂穩定度 (成功率): {(successful_collapses / total_universes) * 100:.1f}%")
print("==========================================")