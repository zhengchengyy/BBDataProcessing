from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 导入全局变量
import GlobalVariable as gv

ndevices = 5
start = 1
end = ndevices

action_names = gv.action_names
feature_names = gv.feature_names

action_names = [
          "turn_over","legs_stretch","hands_stretch",
          "head_move","legs_move","hands_move",
          "kick","legs_tremble","hands_tremble"]

model_list = ["model1","model2","model3","model4","model5"]
X_test_list = ["X_test1","X_test2","X_test3","X_test4","X_test5"]
upper_weight = [2, 1, 2, 2, 1, 2, 1, 1, 2]
lower_weight = [2, 2, 1, 1, 2, 1, 2, 2, 1]
foot_weight = [1, 2, 1, 1, 2, 1, 2, 2, 1]
# foot_weight = [1, 3, 1, 1, 3, 1, 3, 3, 1]
# foot_weight = [1, 1.5, 1, 1, 1.5, 1, 1.5, 1.5, 1]


def loadModel(device_no):
    # 读取模型
    import pickle
    with open('models/' + 'device_' + str(device_no) + '_post_prune.pickle', 'rb') as f:
        model = pickle.load(f)

    return model

def loadMatrix(device_no):
    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')

    # 定义训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, label_matrix, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test

# 测试用
# for device_no in range(start, end + 1):
#     model = loadModel(device_no)
#     X_train, X_test, y_train, y_test = loadMatrix(device_no)
#     train_score = model.score(X_train, y_train)
#     test_score = model.score(X_test, y_test)
#     print('device_' + str(device_no) + '\'s train score:', train_score)
#     print('device_' + str(device_no) +'\'s test score:', test_score)


model1 = loadModel(1)
model2 = loadModel(2)
model3 = loadModel(3)
model4 = loadModel(4)
model5 = loadModel(5)
X_train1, X_test1, y_train1, y_test1 = loadMatrix(1)
X_train2, X_test2, y_train2, y_test2 = loadMatrix(2)
X_train3, X_test3, y_train3, y_test3 = loadMatrix(3)
X_train4, X_test4, y_train4, y_test4 = loadMatrix(4)
X_train5, X_test5, y_train5, y_test5 = loadMatrix(5)

# 测试每个测试集的标签相同
count = 0
for i in range(len(y_test1)):
    if(y_test1[i] != y_test5[i]):
        count += 1
print(count)

# 所有的测试集标签一样
y_test = y_test1
print("------------------加入权重前------------------------")
for device_no in range(start, end+1):
    X_train, X_test, y_train, y_test = loadMatrix(device_no)
    # print("The length of test data",len(y_test))
    train_score = eval("model" + str(device_no) + ".score(X_train, y_train)")
    # print("device_" + str(device_no) + "'train score:", train_score)
    test_score = eval("model"+str(device_no)+".score(X_test, y_test)")
    print("device_"+str(device_no)+"'test score:"+str(test_score)+"≈"+str(round(test_score, 3)))


print("------------------加入权重后------------------------")
def joinWeight(model_list, X_test_list):
    for i in range(len(model_list)):
        result = []
        # pre_list = eval(model_list[i]).predict(eval(X_test_list[i]))
        pre_proba_list = eval(model_list[i]).predict_proba(eval(X_test_list[i]))
        for proba_list in pre_proba_list:
            proba_list_weight = [a * b for a, b in zip(proba_list, foot_weight)]
            idx = np.argmax(proba_list_weight)
            result.append(idx)

        accuracy_acc = 0
        # print(y_test1)
        for j in range(len(result)):
            if (result[j] == y_test[j]):
                accuracy_acc += 1

        score = accuracy_acc / len(y_test)
        print("device_" + str(i + 1) + "'accuracy=" + str(score) + "≈" + str(round(score, 3)))

joinWeight(model_list, X_test_list)