# 移动平均的本质是一种低通滤波。它的目的是过滤掉时间序列中的高频扰动，保留有用的低频趋势。
import numpy as np
import matplotlib.pyplot as plt


def np_move_avg(a, n, mode="same"):
    return (np.convolve(a, np.ones((n,)) / n, mode=mode))


modes = ['full', 'same', 'valid']

print(np.ones((200,)))

for m in modes:
    plt.plot(np_move_avg(np.ones((200,)), 50, mode=m));

plt.axis([-10, 251, -.1, 1.1]);

plt.legend(modes, loc='lower center');

plt.show()