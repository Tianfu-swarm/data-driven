import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# 定义生成函数 C(z)
def C(z, max_k=21):
    result = 0
    for k in range(1, max_k+1):
        term = (k**(k-2) * z**k) / factorial(k)
        result += term
    return result

# 设置 z 的范围
z_values = np.linspace(0, 1, 500)

# 计算 C(z) 对应的值
C_values = np.array([C(z) for z in z_values])

# 绘制图像
plt.figure(figsize=(8, 6))
plt.plot(z_values, C_values, label=r'$C(z) = \sum_{k=1}^{\infty} \frac{k^{k-2} z^k}{k!}$', color='b')
plt.title("Generating Function for Component Size Distribution")
plt.xlabel("z")
plt.ylabel("C(z)")
plt.grid(True)
plt.legend()
plt.show()
