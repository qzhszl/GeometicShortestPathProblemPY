import numpy as np
import matplotlib.pyplot as plt

# 参数设置
beta = 4  # 可以自己改，比如 2, 3, 4, 6
d0_list = [0.05, 0.1, 0.2, 0.4]  # 不同的 d0

# 距离范围
d = np.linspace(0, 1, 500)

# 定义函数
def f(d, d0, beta):
    return 1 / (1 + (d / d0)**beta)

# 画图
plt.figure()

for d0 in d0_list:
    plt.plot(d, f(d, d0, beta), label=f'd0={d0}')

# 图像美化
plt.xlabel('Distance d')
plt.ylabel('Connection probability f(d)')
plt.title(f'SRGG connection function (beta={beta})')
plt.legend()
plt.grid()

plt.show()