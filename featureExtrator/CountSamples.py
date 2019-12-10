import numpy as np
# 导入全局变量
import GlobalVariable as gv

action_names = gv.action_names
feature_names = gv.feature_names

ndevices = 5
start = 1
end = ndevices

def get_count(device_no):
    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')

    count = 0
    pre = 0
    for i in label_matrix:
        if(i == pre):
            count += 1
        else:
            print(action_names[i] + ": " + str(count))
            pre += 1
            count = 0


for i in range(start, end + 1):
    print("--------device_" + str(i) + "--------")
    get_count(i)


