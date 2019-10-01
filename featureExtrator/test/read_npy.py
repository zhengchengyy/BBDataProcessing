import numpy as np

x = np.array([[1,2,3],[4,5,6],[7,8,9]])
a = np.asarray(x)

print(a)

np.save('test.npy', a)

load_array = np.load('test.npy', mmap_mode = 'r+')

print("pre:\n",load_array)

append_array = np.array([0,0,0])

# result = np.r_[load_array,append_array]  #只能将两个矩阵合并
# result = np.insert(load_array,load_array.shape[0], values=append_array, axis=0)

result = np.append(load_array,[append_array],axis=0)

np.save('test.npy', result)

# load_array = np.load('result.npy', mmap_mode = 'r+')

print("post:\n", result)