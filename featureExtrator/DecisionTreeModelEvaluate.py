from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 导入数据
feature_matrix = np.load('feature_matrixs/feature_matrix2.npy')
label_matrix = np.load('feature_matrixs/label_matrix2.npy')

# 定义训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.25, random_state=0)
# print(y_test)

# 读取模型
import pickle

with open('models_discard/0.973Acc_6Fea.pickle', 'rb') as f:
    model = pickle.load(f)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("train score:", train_score)
print("test score:", test_score)

# 模型评估
from sklearn import metrics
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test,y_pred))
#法二：通过混淆矩阵验证（横轴：实际值，纵轴：预测值）（理想情况下是个对角阵）
print(metrics.confusion_matrix(y_test, y_pred))

print("精确率:", metrics.accuracy_score(y_test, y_pred))
print("查准率:", metrics.precision_score(y_test, y_pred, average='macro'))
print("召回率:", metrics.recall_score(y_test, y_pred, average='macro'))
print("F1_score:", metrics.f1_score(y_test, y_pred, average='macro'))

print("kappa:", metrics.cohen_kappa_score(y_test, y_pred))
print("ham_distance:", metrics.hamming_loss(y_test, y_pred))
# print("jaccrd_score:", metrics.jaccard_similarity_score(y_test, y_pred))
# print("hinger:", metrics.hinge_loss(y_test, y_pred))

# 交叉验证
from sklearn.model_selection import cross_val_score
print("交叉验证:", cross_val_score(model, X_train, y=y_train, cv=3))  # cv表示几倍交叉验证
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