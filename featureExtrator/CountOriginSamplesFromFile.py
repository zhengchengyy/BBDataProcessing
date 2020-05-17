import numpy as np

origin_num = np.load('origin_num.npy')
origin_num = origin_num.tolist()
for i in range(1, len(origin_num) + 1):
    print(origin_num[i])


for i in range(1, len(origin_num) + 1):
    eight_people = 0
    two_people = 0
    for j in range(len(origin_num[i])):
        for k in range(len(origin_num[i][j])):
            if(k < 8):
                eight_people += origin_num[i][j][k]
            else:
                two_people += origin_num[i][j][k]
    print(f"----------device_{i}----------")
    print(f"the sum is {eight_people+two_people}.")
    print(f"eight people data include {eight_people}.")
    print(f"two people data include {two_people}.")