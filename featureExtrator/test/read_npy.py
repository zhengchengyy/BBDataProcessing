import numpy as np

x =  [1,2,3]
a = np.asarray(x)

print(a)

np.save('test', a)

test = np.load('test.npy')

print(test)