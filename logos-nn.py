import random
import math

class LogosLayerSoftplus:
    def __init__(self, in_dim, out_dim, lr=0.1, threshold=2.0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.threshold = threshold
        
        # R+ 權重初始化
        self.W_pos = [[random.uniform(0.1, 0.5) for _ in range(in_dim)] for _ in range(out_dim)]
        self.W_neg = [[random.uniform(0.1, 0.5) for _ in range(in_dim)] for _ in range(out_dim)]
        
        # Hadamard 雙軌遮罩 (對稱且正向)
        self.H_pos = [random.choice([0.5, 1.5]) for _ in range(in_dim)]
        self.H_neg = [2.0 - m for m in self.H_pos]

    def _softplus(self, x):
        # f(x) = ln(1 + exp(x))
        # 為了防止數值溢位，當 x 很大時，f(x) 趨近於 x
        if x > 20: return x
        return math.log(1 + math.exp(x))

    def forward_and_update(self, x_input, is_positive=True):
        # 1. 雙軌 Hadamard 散射
        x_p = [val * mask for val, mask in zip(x_input, self.H_pos)]
        x_n = [val * mask for val, mask in zip(x_input, self.H_neg)]
        
        # 2. 正向傳導：使用 Softplus 進行差動激發
        h = []
        for i in range(self.out_dim):
            excite = sum(self.W_pos[i][j] * x_p[j] for j in range(self.in_dim))
            inhib = sum(self.W_neg[i][j] * x_n[j] for j in range(self.in_dim))
            # Softplus 確保輸出永遠 > 0
            h.append(self._softplus(excite - inhib))
        
        # 3. 計算局部 Goodness
        goodness = sum(val**2 for val in h) / len(h)
        
        # 4. 即時局部更新
        diff = self.threshold - goodness
        direction = 1.0 if is_positive else -1.0
        update_scale = direction * diff
        
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                # 更新雙軌
                self.W_pos[i][j] += self.lr * update_scale * h[i] * x_p[j]
                self.W_neg[i][j] -= self.lr * update_scale * h[i] * x_n[j]
                
                # 嚴格守護 R+
                if self.W_pos[i][j] < 1e-5: self.W_pos[i][j] = 1e-5
                if self.W_neg[i][j] < 1e-5: self.W_neg[i][j] = 1e-5
        
        norm = math.sqrt(sum(val**2 for val in h)) + 1e-8
        h_normed = [v / norm for v in h]
        return h_normed, goodness

# --- 實驗環節 ---
pos_samples = [[0,0, 0,1], [0,1, 1,1], [1,0, 1,1], [1,1, 0,1]] # [x1, x2, label, bias]
neg_samples = [[0,0, 1,1], [0,1, 0,1], [1,0, 0,1], [1,1, 1,1]]

l1 = LogosLayerSoftplus(4, 24, lr=0.1)
l2 = LogosLayerSoftplus(24, 12, lr=0.1)

print("開始執行 logos-nn v6 (Softplus 演化版)...")

for epoch in range(4000):
    for p, n in zip(pos_samples, neg_samples):
        h1_p, _ = l1.forward_and_update(p, is_positive=True)
        h1_n, _ = l1.forward_and_update(n, is_positive=False)
        l2.forward_and_update(h1_p, is_positive=True)
        l2.forward_and_update(h1_n, is_positive=False)

print("訓練完成。推論測試：")
for x in [[0,0], [0,1], [1,0], [1,1]]:
    h1_0, g1_0 = l1.forward_and_update([x[0], x[1], 0, 1], is_positive=True)
    _, g2_0 = l2.forward_and_update(h1_0, is_positive=True)
    
    h1_1, g1_1 = l1.forward_and_update([x[0], x[1], 1, 1], is_positive=True)
    _, g2_1 = l2.forward_and_update(h1_1, is_positive=True)
    
    res0, res1 = g1_0 + g2_0, g1_1 + g2_1
    print(f"XOR({x[0]}, {x[1]}) -> 活躍度(假設0): {res0:4.2f} | 活躍度(假設1): {res1:4.2f} | 預測: {0 if res0 > res1 else 1}")