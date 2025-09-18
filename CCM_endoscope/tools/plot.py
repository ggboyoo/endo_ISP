import numpy as np
import matplotlib.pyplot as plt

# 数据（g 为横坐标）
g = np.array([13, 12, 11, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
y = np.array([5.5789, 5.131096, 4.832, 4.2, 3.839, 3.53, 3.1057, 2.71, 2.31, 1.755, 1.48, 1.005], dtype=float)

# 线性拟合 y = a*g + b
a, b = np.polyfit(g, y, 1)
y_fit = a * g + b

# R^2
ss_res = np.sum((y - y_fit)**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - ss_res / ss_tot

# 绘图
plt.figure(figsize=(6,4))
plt.scatter(g, y, color='blue', label='Data')
# 为了直线更平滑，按范围画
gx = np.linspace(g.min(), g.max(), 200)
plt.plot(gx, a*gx + b, color='red', label=f'Fit: y={a:.4f}·g+{b:.4f}\nR²={r2:.4f}')
plt.xlabel('g')
plt.ylabel('y')
plt.title('Linear Fit (y vs g)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('fit_y_vs_g.png', dpi=200)
plt.show()

print(f'a={a:.6f}, b={b:.6f}, R^2={r2:.6f}')