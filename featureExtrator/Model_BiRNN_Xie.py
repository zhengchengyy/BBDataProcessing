from sklearn.model_selection import train_test_split
import numpy as np


# 导入全局变量
import GlobalVariable as gv
action_names = gv.action_names
feature_names = gv.feature_names
# 计算每类准确率函数
def compute_accuracy(predict_result, true_result):
    sum = 0
    for i in range(len(action_names)):
        print(action_names[i], end=": ")
        count = 0
        correct_count = 0
        for j in range(len(predict_result)):
            if(i == predict_result[j]):
                count += 1
                if(predict_result[j] == true_result[j]):
                    correct_count += 1
        if (count != 0):
            correct = correct_count / count
            sum += correct
            print(str(correct) +"≈" + str(round(correct, 3)),end="")
            print(str("(") + str(correct_count) + ":" + str(count) + str(")"), end="=")
            print(str(round(correct * 100, 1)) + "%")
        else:
            print("不存在该类")

    average_accuracy = sum / len(action_names)
    print("平均识别准确率: " + str(average_accuracy), end=" ")
    print(str(round(average_accuracy * 100, 1)) + "%")


# 导入数据
device_no = 1
feature_matrix = np.load('feature_matrixs/feature_matrix' + str(device_no) + '.npy')
label_matrix = np.load('feature_matrixs/label_matrix' + str(device_no) + '.npy')
# print(feature_matrix.shape)  #(2830, 2)
# print(label_matrix.shape)  #(2830,)

# 定义训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_matrix, test_size=0.2, random_state=1)
# print(X_train.shape)  #(2264, 2)
# print(X_test.shape)  #(566, 2)

# 标准化，均值为0，方差为1
from sklearn import preprocessing
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

from keras.models import Sequential
from keras.layers import LSTM,Dense,Bidirectional
from keras.utils import to_categorical

# 将类别向量转换为二进制(只有0和1)的矩阵类型表示
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


model=Sequential()
# model.add(Bidirectional(LSTM(input_shape=(1,2),units=16),batch_input_shape=(None,1,2)))
model.add(Bidirectional(LSTM(input_shape=(1,2),units=16),input_shape=(1,2)))
# model.add(LSTM(input_shape=(1,2),units=16))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(y_train.shape[1],activation='softmax'))  #y_train.shape[1]=class num

X_train=X_train.reshape((-1,1,2))
X_test=X_test.reshape((-1,1,2))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200, batch_size=64)


# 评估测试数据的准确率
print('\nTesting ------------')
# Evaluate the models_discard with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)


# 预测，得到每类的准确率
predict_result = model.predict(X_test)
# print(predict_result.shape)  #(566, 9)
# 把二进制矩阵类型表示转换类别向量，例如[0,0,1]->[2]
y_test_class = []
for i in y_test:
    y_test_class.append(np.argmax(i))
predict_result_class = []
for i in predict_result:
    predict_result_class.append(np.argmax(i))
compute_accuracy(predict_result_class, y_test_class)


# 使用DNN，Dense表示全连接层
# model=Sequential()
# model.add(Dense(input_dim=2,units=16,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(y_train.shape[1],activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])