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

    return X_train, X_test, y_train, y_test


model1 = loadModel(1,0.905)
model2 = loadModel(2,0.924)
model3 = loadModel(3,0.973)
X_train1, X_test1, y_train1, y_test1 = loadMatrix(1)
X_train2, X_test2, y_train2, y_test2 = loadMatrix(2)
X_train3, X_test3, y_train3, y_test3 = loadMatrix(3)
y_test = y_test1

for device_no in range(1,3+1):
    X_train, X_test, y_train, y_test = loadMatrix(device_no)
    # print("The length of test data",len(y_test))
    train_score = eval("model" + str(device_no) + ".score(X_train, y_train)")
    # print("device_" + str(device_no) + "'train score:", train_score)
    test_score = eval("model"+str(device_no)+".score(X_test, y_test)")
    print("device_"+str(device_no)+"'test score:", test_score)


print("------------------规则1：不同推理机动作结论概率最大值为最终结论的准确率---------------------------")
# 多个设备时
model_list = ["model1","model2","model3"]
X_test_list = ["X_test1","X_test2","X_test3"]
def combine(model_list, X_test_list):
    result = []
    action_proba_list = []
    com_name = ""
    for i in range(len(model_list)):
        pre_list = eval(model_list[i]).predict(eval(X_test_list[i]))
        pre_proba_list = eval(model_list[i]).predict_proba(eval(X_test_list[i]))

        # 定义模型组合名字
        if(i==len(model_list)-1):
            com_name += model_list[i][-1]
        else:
            com_name += model_list[i][-1] + "+"

        action_proba = []
        for j in range(len(pre_list)):
            pre = pre_list[j]
            pre_proba = pre_proba_list[j][pre]
            action_proba.append([pre,pre_proba])
        print(action_proba)
        action_proba_list.append(action_proba)
    for i in range(len(action_proba_list[0])):
        max_proba = 0
        max_action = 0
        for j in range(len(action_proba_list)):
            action = action_proba_list[j][i][0]
            proba = action_proba_list[j][i][1]
            if(proba>=max_proba):
                max_proba = proba
                max_action = action
        result.append(max_action)
    accuracy_acc = 0
    for i in range(len(result)):
        if (result[i] == y_test[i]):
            accuracy_acc += 1

    print("device("+ com_name + ")'accuracy=" + str(accuracy_acc / len(result)))


combine(["model1","model2","model3"], ["X_test1","X_test2","X_test3"])
combine(["model1","model2"], ["X_test1","X_test2"])
combine(["model1","model3"], ["X_test1","X_test3"])
combine(["model2","model3"], ["X_test2","X_test3"])


print("------------------规则2：不同推理机动作结论概率和的最大值为最终结论的准确率------------------------")
# 多个设备时
model_list = ["model1","model2","model3"]
X_test_list = ["X_test1","X_test2","X_test3"]
def combine2(model_list, X_test_list):
    result = []
    action_proba_list = []
    com_name = ""
    for i in range(len(model_list)):
        pre_list = eval(model_list[i]).predict(eval(X_test_list[i]))
        pre_proba_list = eval(model_list[i]).predict_proba(eval(X_test_list[i]))

        # 定义模型组合名字
        if(i==len(model_list)-1):
            com_name += model_list[i][-1]
        else:
            com_name += model_list[i][-1] + "+"

        action_proba = []
        for j in range(len(pre_list)):
            pre = pre_list[j]
            pre_proba = pre_proba_list[j][pre]
            action_proba.append([pre,pre_proba])
        action_proba_list.append(action_proba)
    for i in range(len(action_proba_list[0])):
        max_proba = 0
        max_action = 0
        action_dic = {}
        for j in range(len(action_proba_list)):
            action = action_proba_list[j][i][0]
            proba = action_proba_list[j][i][1]
            if action in action_dic:
                action_dic[action] += proba
            else:
                action_dic[action] = proba
        for k in action_dic.keys():
            if (proba >= max_proba):
                max_proba = proba
                max_action = action
        result.append(max_action)
    accuracy_acc = 0
    for i in range(len(result)):
        if (result[i] == y_test[i]):
            accuracy_acc += 1

    print("device("+ com_name + ")'accuracy=" + str(accuracy_acc / len(result)))


combine2(["model1","model2","model3"], ["X_test1","X_test2","X_test3"])
combine2(["model1","model2"], ["X_test1","X_test2"])
combine2(["model1","model3"], ["X_test1","X_test3"])
combine2(["model2","model3"], ["X_test2","X_test3"])


print("------------------测试：以下用简单方法测试上述代码准确性----------------------------")
# 两个设备时，可以用于测试
def combine_test(com_name, model1,model2,X_test1,X_test2):
    result = []
    pre_list1 = model1.predict(X_test1)
    pre_list2 = model2.predict(X_test2)
    pre_proba_list1 = model1.predict_proba(X_test1)
    pre_proba_list2 = model2.predict_proba(X_test2)
    for i in range(len(pre_list1)):
        pre1 = pre_list1[i]
        pre2 = pre_list2[i]
        pre_proba1 = pre_proba_list1[i][pre1]
        pre_proba2 = pre_proba_list2[i][pre2]
        if(pre1!=pre2):
            if(pre_proba1>pre_proba2):
                result.append(pre1)
            else:
                result.append(pre2)
        else:
            result.append(pre1)
    # print(result)
    # print(y_test)
    accuracy_acc = 0
    for i in range(len(result)):
        if(result[i]==y_test[i]):
            accuracy_acc += 1
    print(com_name + "'accuracy=" + str(accuracy_acc/len(y_test)))

combine_test("device(1+2)",model1,model2,X_test1,X_test2)
combine_test("device(1+3)",model1,model3,X_test1,X_test3)
combine_test("device(2+3)",model2,model3,X_test2,X_test3)
