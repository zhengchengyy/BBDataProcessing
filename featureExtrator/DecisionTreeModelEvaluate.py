from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 导入数据
device_no = 1
feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')

# 导入全局变量
import GlobalVariable as gv
action_names = gv.action_names
feature_names = gv.feature_names

# 定义训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.2, random_state=0)
# print(len(y_test))

# 读取模型
import pickle

# with open('models/device_1Acc_0.887Fea_2.pickle', 'rb') as f:
with open('models/' + 'device_' + str(device_no) + '_post_prune.pickle', 'rb') as f:
    model = pickle.load(f)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("train score:", train_score)
print("test score:", test_score)

# 模型评估
from sklearn import metrics
y_pred = model.predict(X_test)
# 微平均值：micro average，所有数据结果的平均值
# 宏平均值：macro average，所有标签结果的平均值
# 加权平均值：weighted average，所有标签结果的加权平均值
print(metrics.classification_report(y_test, y_pred, digits=4, target_names=action_names))
# print(metrics.balanced_accuracy_score(y_test,y_pred))  # 数据不平衡情况下
#法二：通过混淆矩阵验证（横轴：实际值，纵轴：预测值）（理想情况下是个对角阵）
matrix = metrics.confusion_matrix(y_test, y_pred)
print("混淆矩阵：\n", matrix)
accuracy_everyclass = matrix.diagonal()/matrix.sum(axis=1)
print("每一类的准确率:", accuracy_everyclass)

print("准确率:", metrics.accuracy_score(y_test, y_pred))
print("宏精确率:", metrics.precision_score(y_test, y_pred, average='macro'))
print("微精确率:", metrics.precision_score(y_test, y_pred, average='micro'))

print("宏召回率:", metrics.recall_score(y_test, y_pred, average='macro'))
print("微召回率:", metrics.recall_score(y_test, y_pred, average='micro'))

print("宏F1_score:", metrics.f1_score(y_test, y_pred, average='macro'))
print("微F1_score:", metrics.f1_score(y_test, y_pred, average='micro'))

print("kappa:", metrics.cohen_kappa_score(y_test, y_pred))
print("ham_distance:", metrics.hamming_loss(y_test, y_pred))
# print("jaccrd_score:", metrics.jaccard_similarity_score(y_test, y_pred))
# print("hinger:", metrics.hinge_loss(y_test, y_pred))

# 交叉验证
from sklearn.model_selection import cross_val_score
print("交叉验证分数:", cross_val_score(model, X_train, y=y_train, cv=3))  # cv表示几倍交叉验证
print("交叉验证平均分数:", cross_val_score(model, X_train, y=y_train, cv=3).mean())
from sklearn.model_selection import validation_curve
# 检验曲线，传入不同参数和对应参数范围
train_score, test_score = validation_curve(model, X_train, y_train,
                        "max_depth", range(1,10), cv=5, scoring=None, n_jobs=1)
# print(train_score)

# loss损失
pred = model.predict_proba(X_test)
# print(pred)
# print(y_test)
print("loss损失:", metrics.log_loss(y_test, pred))