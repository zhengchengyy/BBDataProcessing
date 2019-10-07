import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

t = np.linspace(0, 1, 400, endpoint=False)
cond = [t<0.25, (t>=0.25)&(t<0.5), t>=0.5]
f1 = lambda t: np.cos(2*np.pi*10*t)
f2 = lambda t: np.cos(2*np.pi*50*t)
f3 = lambda t: np.cos(2*np.pi*100*t)

y1 = np.piecewise(t, cond, [f1, f2, f3])
y2 = np.piecewise(t, cond, [f2, f1, f3])

Y1 = abs(fft(y1))
Y2 = abs(fft(y2))

plt.figure(figsize=(12, 9))
plt.subplot(221)
plt.plot(t, y1)
plt.title('signal_1 in time domain')
plt.xlabel('Time/second')

from scipy import signal
b, a = signal.butter(8, 100*2/400, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
filtedData = signal.filtfilt(b, a, Y1)  #data为要过滤的信号

plt.subplot(222)
plt.plot(range(400), filtedData)
plt.title('signal_1 in frequency domain')
plt.xlabel('Frequency/Hz')

plt.subplot(223)
plt.plot(t, y2)
plt.title('signal_2 in time domain')
plt.xlabel('Time/second')

plt.subplot(224)
plt.plot(range(400), Y2)
plt.title('signal_2 in frequency domain')
plt.xlabel('Frequency/Hz')

plt.tight_layout()
plt.show()