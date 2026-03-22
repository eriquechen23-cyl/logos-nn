import numpy as np
import sys

class LogosLayerNumPyXOR:
    def __init__(self, in_dim, out_dim, lr=0.05, threshold=3.0, beta=0.1):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.threshold = threshold
        self.beta = beta
        
        scale = np.sqrt(2.0 / in_dim) 
        self.W_pos = np.random.uniform(0.1, scale, (out_dim, in_dim))
        self.W_neg = np.random.uniform(0.1, scale, (out_dim, in_dim))
        
        self.H_pos = np.random.choice([0.8, 1.2], size=in_dim)
        self.H_neg = 2.0 - self.H_pos

    def _shifted_softplus(self, x):
        return np.log1p(np.exp(np.clip(x - self.beta, -15, 15)))

    def forward_and_update(self, x_input, is_positive=True, update=True):
        x_norm = x_input / (np.linalg.norm(x_input) + 1e-8)
        
        x_p = x_norm * self.H_pos
        x_n = x_norm * self.H_neg
        
        excite = self.W_pos @ x_p
        inhib = self.W_neg @ x_n
        h = self._shifted_softplus(excite - inhib)
        
        goodness = np.mean(h**2)
        
        if update:
            p_truth = 1.0 / (1.0 + np.exp(-(goodness - self.threshold)))
            
            if is_positive:
                drive = self.lr * (1.0 - p_truth)
                self.W_pos += drive * np.outer(h, x_p)
                self.W_neg -= drive * np.outer(h, x_n)
            else:
                drive = self.lr * p_truth 
                self.W_neg += drive * np.outer(h, x_n)
                self.W_pos -= drive * np.outer(h, x_p)
            
            self.W_pos = np.clip(self.W_pos, 1e-4, 5.0) 
            self.W_neg = np.clip(self.W_neg, 1e-4, 5.0)
            
        return h, goodness

# ==========================================
# 實驗環節：100 個平行宇宙壓力測試
# ==========================================

pos_samples = np.array([[0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0]], dtype=np.float32)
neg_samples = np.array([[0, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1]], dtype=np.float32)

num_seeds = 100
perfect_runs = 0
failed_seeds = []

print(f"⚡ 開始執行 {num_seeds} 個平行宇宙的穩定度驗證...\n")

for current_seed in range(num_seeds):
    # 鎖定當前宇宙的創世種子
    np.random.seed(current_seed)
    
    # 初始化網路
    l1 = LogosLayerNumPyXOR(4, 64, lr=0.05, threshold=3.0, beta=0.1)
    l2 = LogosLayerNumPyXOR(64, 16, lr=0.05, threshold=3.0, beta=0.1)
    
    # 訓練 2000 Epoch
    indices = np.arange(4)
    for epoch in range(2000):
        np.random.shuffle(indices)
        for i in indices:
            h1_p, _ = l1.forward_and_update(pos_samples[i], is_positive=True)
            l2.forward_and_update(h1_p, is_positive=True)
            
            h1_n, _ = l1.forward_and_update(neg_samples[i], is_positive=False)
            l2.forward_and_update(h1_n, is_positive=False)

    # 推論測試
    success_count = 0
    for x1, x2, true_label in [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]:
        input_0 = np.array([x1, x2, 1, 0], dtype=np.float32)
        h1_0, g1_0 = l1.forward_and_update(input_0, update=False)
        _, g2_0 = l2.forward_and_update(h1_0, update=False)
        
        input_1 = np.array([x1, x2, 0, 1], dtype=np.float32)
        h1_1, g1_1 = l1.forward_and_update(input_1, update=False)
        _, g2_1 = l2.forward_and_update(h1_1, update=False)
        
        res0 = g1_0 + g2_0
        res1 = g1_1 + g2_1
        prediction = 0 if res0 > res1 else 1
        
        if prediction == true_label:
            success_count += 1
            
    # 統計並輸出視覺化進度
    if success_count == 4:
        perfect_runs += 1
        print("✅", end="", flush=True)
    else:
        failed_seeds.append((current_seed, success_count))
        print("❌", end="", flush=True)
        
    # 每 20 個換行，方便閱讀
    if (current_seed + 1) % 20 == 0:
        print(f" [{current_seed + 1}/{num_seeds}]")

print(f"\n\n🎯 驗證完成！")
print(f"🌟 完美通關 (4題全對) 的宇宙數量: {perfect_runs} / {num_seeds} ({perfect_runs/num_seeds*100:.1f}%)")

if failed_seeds:
    print(f"\n⚠️ 陷入幾何死角的種子 (Seed, 答對題數):")
    print(failed_seeds)
else:
    print("\n🏆 不可思議！Logos-NN 在所有平行宇宙中皆尋得真理！")