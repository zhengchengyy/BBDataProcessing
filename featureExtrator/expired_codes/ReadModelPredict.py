# 读取模型
start = 1
end = start
device_no = 1
import pickle
with open('models/' + 'device_' + str(device_no) + '_model.pickle', 'rb') as f:
    model = pickle.load(f)

feature_app = [0.58061192, 0.04920815]
feature_app = [0.58229542, 0.05709379]
feature_app = [0.5806119165561056, 0.049208149271103105]
feature_app = [0.5806119165561056, 0.049208149271103105]
predict_result = model.predict([feature_app])
print(predict_result)