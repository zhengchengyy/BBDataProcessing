import pywt
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(512)
# y = np.sin(2*np.pi*x/32)
y = np.array([1]*512)
coef, freqs=pywt.cwt(y,np.arange(1,129),'cgau8')
plt.subplot(411)
# plt.matshow(coef) # doctest: +SKIP
plt.plot(x,y)
plt.subplot(412)
plt.contourf(x, freqs, abs(coef))

import pywt
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
plt.subplot(413)
plt.plot(t,sig)
widths = np.arange(1, 31)
cwtmatr, freqs = pywt.cwt(sig, widths, 'cgau8')
plt.subplot(414)
plt.contourf(t, freqs, abs(cwtmatr))
# plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # doctest: +SKIP
plt.show() # doctest: +SKIP