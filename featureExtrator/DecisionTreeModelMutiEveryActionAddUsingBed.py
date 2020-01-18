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
          "kick","legs_tremble","hands_tremble",
          "go_to_bed", "get_up"]
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
print("------------------规则1：不同推理机动作结论概率最大值为最终结论的准确率---------------------------")
def combine(model_list, X_test_list):
    result = []
    action_proba_list = []
    com_name = ""
    for i in range(len(model_list)):
        pre_list = eval(model_list[i]).predict(eval(X_test_list[i]))
        pre_proba_list = eval(model_list[i]).predict_proba(eval(X_test_list[i]))

        # pre_bed_list = eval(model_bed_list[i]).predict(eval(X_test_bed_list[i]))
        # pre_proba_bed_list = eval(model_bed_list[i]).predict_proba(eval(X_test_bed_list[i]))

        # 定义模型组合名字
        if(i==len(model_list)-1):
            com_name += model_list[i][-1]
        else:
            com_name += model_list[i][-1] + "+"

        action_proba = []
        for j in range(len(pre_list)):
            pre = pre_list[j]
            if (pre >= 9):
                pre -= 9
            pre_proba = pre_proba_list[j][pre]
            pre += 9
            action_proba.append([pre,pre_proba])
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
    score = accuracy_acc / len(y_test)
    print(com_name + "'accuracy=" + str(score) + "≈" + str(round(score, 3)))

    for i in range(len(action_names)):
        if(i < 9):
            continue
        print(action_names[i], end=":")
        count = 0
        correct_count = 0
        for j in range(len(result)):
            if(i == result[j]):
                count += 1
                if (result[j] == y_test[j]):
                    correct_count += 1
        if(count != 0):
            print(correct_count / count, end="")
        print(str("(") + str(correct_count) + ":" + str(count) + str(")"))

print("9种常见动作测试数据个数：", len(y_test))
print("上下床测试数据个数：", len(y_test_bed))
print(y_test_bed)
# 使用上下床规则来识别9种动作，观察是否会误识别为上下床动作，同样会误识别为这2种动作，但是当有这9种动作的规则时，
# 可以大概率把这9种动作识别出来，不会把9种动作数据误识别为上下床
# 因为我们可以把上下床规则添加到规则集的最后，匹配时上下床规则优先级更高，这时候更容易误识别呀？
# 不过假设其它动作和上下床动作差异最明显的情况下，只有9种动作规则时会把上下床误识别为这9种动作
# 当增加上下床规则后，由于上下床识别的准确性，如果要保证上下床的识别率为100%，此时其它动作不能误识别上下床动作
# 上下床误识别为这9种动作，但是这9种动作也可能误识别为这2种动作，如何解决？
# 这时看单个动作，上下床规则增加前后会导致识别率的增加还是下降？不能确定，但能确定上下床动作的识别率肯定不是100%
combine(model_bed_list, X_test_list)