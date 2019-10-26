import numpy as np
import matplotlib.pyplot as plt

tang_array = [np.random.uniform(0, std, 100) for std in [0.1, 0.2, 0.3, 0.4]]
tang_array =[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
bar_labels = ['x1', 'x2', 'x3', 'x4']
print(tang_array)

fig = plt.figure()
plt.xticks([x+1 for x in range(len(tang_array))], bar_labels)
bplt = plt.boxplot(tang_array, notch=False, sym='o', vert=True, patch_artist=True)

colors = ['pink', 'lightblue', 'lightgreen']
for pacthes, color in zip(bplt['boxes'], colors):
    pacthes.set_facecolor(color)

plt.show()