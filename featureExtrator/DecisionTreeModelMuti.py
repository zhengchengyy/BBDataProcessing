from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 导入全局变量
import GlobalVariable as gv

action_names = gv.action_names
feature_names = gv.feature_names
feature_num = len(feature_names)


def loadModel(device_no, test_score):
    # 读取模型
    import pickle
    with open('models/' + 'device_' + str(device_no) + 'Acc_' + str(round(test_score, 3))
              + 'Fea_' + str(feature_num) + '_prune.pickle', 'rb') as f:
        model = pickle.load(f)

    return model


def loadMatrix(device_no):
    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')

    # 定义训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, label_matrix, test_size=0.25, random_state=0)

    return X_test, y_test


model1 = loadModel(1,0.905)
model2 = loadModel(2,0.924)
model3 = loadModel(3,0.973)
X_test1,y_test1 = loadMatrix(1)
X_test2,y_test2 = loadMatrix(2)
X_test3,y_test3 = loadMatrix(3)


for device_no in range(1,3+1):
    X_test, y_test = loadMatrix(device_no)
    print("The length of test data",len(y_test))
    # print(y_test)
    test_score = test_score = eval("model"+str(device_no)+".score(X_test, y_test)")
    # print("device_"+str(device_no)+"'test score:", test_score)

# 1号设备+2号设备
pre1 = model1.predict(X_test)
pre2 = model2.predict(X_test)
print(pre1)
print(pre2)
accuracy_acc = 0
for i in range(len(pre1)):
    if(pre1[i]==pre2[i] and pre1[i]==y_test[i]):
        accuracy_acc += 1
print(accuracy_acc/len(pre1))

# for i in range(length):
#     pre1 = model1.predict(X_test[i])
#     pre2 = model2.predict(X_test[i])
#     if(pre1 == pre2):
#         result.append(pre1)
# accuracy_acc = 0
# for i in range(length):
#     if(result[i]==y_test[i]):
#         accuracy_acc += 1
# print(accuracy_acc/length)
