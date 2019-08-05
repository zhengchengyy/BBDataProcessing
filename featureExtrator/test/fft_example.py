# 在这个例子中，实数输入有一个Hermitian的FFT，即在实部中是对称的，在虚部中是反对称的，如numpy.fft文档中所述：
# https://numpy.org/devdocs/reference/routines.fft.html#module-numpy.fft
# https://numpy.org/devdocs/reference/generated/numpy.fft.fft.html#numpy.fft.fft

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(256)
print(t.shape[-1])
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1])
plt.plot(freq, sp.real)  # 图像是对称的
plt.plot(freq, sp.imag)  # 图像是反对称的

plt.show()
