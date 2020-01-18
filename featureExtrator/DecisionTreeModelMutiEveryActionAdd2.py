from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 导入全局变量
import GlobalVariable as gv

ndevices = 5
start = 1
end = ndevices

action_names = gv.action_names
action_names = [
          "turn_over","legs_stretch","hands_stretch",
          "head_move","legs_move","hands_move",
          "kick","legs_tremble","hands_tremble"]
feature_names = gv.feature_names

model_list = ["model1","model2","model3","model4","model5"]
X_test_list = ["X_test1","X_test2","X_test3","X_test4","X_test5"]
model_bed_list = ["model_bed1","model_bed2","model_bed3","model_bed4","model_bed5"]
X_test_bed_list = ["X_test_bed1","X_test_bed2","X_test_bed3","X_test_bed4","X_test_bed5"]

def loadModel(device_no):
    # 读取模型
    import pickle
    with open('models/' + 'device_' + str(device_no) + '_post_prune.pickle', 'rb') as f:
        model = pickle.load(f)

    return model

def loadModelBed(device_no):
    # 读取模型
    import pickle
    with open('models/' + 'device_' + str(device_no) + '_bed_post_prune.pickle', 'rb') as f:
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

def loadMatrixBed(device_no):
    # 导入数据
    feature_matrix = np.load('feature_matrixs/feature_matrix_bed' + str(device_no) + '.npy')
    label_matrix = np.load('feature_matrixs/label_matrix_bed' + str(device_no) + '.npy')

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

model_bed1 = loadModelBed(1)
model_bed2 = loadModelBed(2)
model_bed3 = loadModelBed(3)
model_bed4 = loadModelBed(4)
model_bed5 = loadModelBed(5)
X_train_bed1, X_test_bed1, y_train_bed1, y_test_bed1 = loadMatrixBed(1)
X_train_bed2, X_test_bed2, y_train_bed2, y_test_bed2 = loadMatrixBed(2)
X_train_bed3, X_test_bed3, y_train_bed3, y_test_bed3 = loadMatrixBed(3)
X_train_bed4, X_test_bed4, y_train_bed4, y_test_bed4 = loadMatrixBed(4)
X_train_bed5, X_test_bed5, y_train_bed5, y_test_bed5 = loadMatrixBed(5)

for device_no in range(start, end+1):
    X_train, X_test, y_train, y_test = loadMatrix(device_no)
    # print("The length of test data",len(y_test))
    train_score = eval("model" + str(device_no) + ".score(X_train, y_train)")
    # print("device_" + str(device_no) + "'train score:", train_score)
    test_score = eval("model"+str(device_no)+".score(X_test, y_test)")
    print("device_"+str(device_no)+"'test score:"+str(test_score)+"≈"+str(round(test_score, 3)))

for device_no in range(start, end + 1):
    X_train_bed, X_test_bed, y_train_bed, y_test_bed = loadMatrixBed(device_no)
    # print("The length of test data",len(y_test))
    train_score = eval("model_bed" + str(device_no) + ".score(X_train_bed, y_train_bed)")
    # print("device_" + str(device_no) + "_bed'train score:", train_score)
    test_score = eval("model_bed" + str(device_no) + ".score(X_test_bed, y_test_bed)")
    print("device_" + str(device_no) + "_bed'test score:" + str(test_score) + "≈" + str(round(test_score, 3)))

# 所有的测试集标签一样
y_test = y_test1
y_test_bed = y_test_bed1
print("------------------规则2：不同推理机动作结论概率和的最大值为最终结论的准确率------------------------")
def combine2(model_list, X_test_list):
    result = []
    action_proba_list = []
    com_name = ""
    for i in range(len(model_list)):
        pre_list = eval(model_list[i]).predict(eval(X_test_list[i]))
        pre_proba_list = eval(model_list[i]).predict_proba(eval(X_test_list[i]))

        # 定义模型组合名字
        if (i == len(model_list) - 1):
            com_name += model_list[i][-1]
        else:
            com_name += model_list[i][-1] + "+"

        action_proba = []
        for j in range(len(pre_list)):
            pre = pre_list[j]
            pre_proba = pre_proba_list[j][pre]
            action_proba.append([pre, pre_proba])
        action_proba_list.append(action_proba)

    # action_proba_list为[[[1,2],[2,3],[1,1]],……],shape=(5, 566, 2)
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

        for key in action_dic.keys():
            if (action_dic[key] >= max_proba):
                max_proba = action_dic[key]
                max_action = key
        result.append(max_action)
    accuracy_acc = 0
    for i in range(len(result)):
        if (result[i] == y_test[i]):
            accuracy_acc += 1

    score = accuracy_acc / len(y_test)
    print(com_name + "'accuracy=" + str(score) + "≈" + str(round(score, 3)))

    for i in range(len(action_names)):
        print(action_names[i], end=":")
        count = 0
        correct_count = 0
        for j in range(len(result)):
            if(i == result[j]):
                count += 1
                if (result[j] == y_test[j]):
                    correct_count += 1
        if(count != 0):
            correct = correct_count / count
            print(str(correct) + "≈" + str(round(correct, 3)), end="")
        print(str("(") + str(correct_count) + ":" + str(count) + str(")"))

print("9种常见动作测试数据个数：", len(y_test))
print("上下床测试数据个数：", len(y_test_bed))
print("---------计算没有上下床数据的识别率---------")
combine2(model_list, X_test_list)

# 在只有9中动作规则时会把上下床动作的数据误判为这9种动作之一
# 当增加上下床规则后，会把这上下床动作识别正确，也有可能会识别错误，把其识别成其它9种动作，但是其概率小
# 可以把这部分忽略，因为本文把上下床动作规则添加进规则库的后面，按照匹配方式从前往后，这样使其被误识别为其它动作的概率更小
print("---------计算上下床数据被误识别为其它动作的个数---------")
combine2(model_list, X_test_bed_list)
# 得到的结果理应分子为0，但是有得1的数字，原因是不应该比较result[j] == y_test[j]，而应result[j] == y_test_bed[j]
# 所以这里分子为1毫不意义，瞎猫撞上死耗子

# combine(model_bed_list, X_test_bed_list)
