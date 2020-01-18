# 实时准确率为什么很多人的动作为0
from feature_modules import *
from feature_extractor import FeatureExtractor
import time

from pymongo import MongoClient
from exceptions import CollectionError
from pymongo import MongoClient

from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import os

config = {'db': 'beaglebone',
          'tag_collection': 'tags_411',
          'volt_collection': 'volts_411',
          'offset': 0}

config = {'db': 'beaglebone',
          'tag_collection': 'tags_1105',
          'volt_collection': 'volts_1105',
          'device_num': 5,
          'offset': 0,
          'interval': 2,
          'rate': 1}


# 导入全局变量
import GlobalVariable as gv
action_names = gv.action_names
feature_names = gv.feature_names


def timeToFormat(t):
    ftime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    return ftime


def timeToSecond(t):
    stime = time.strftime("%M:%S", time.localtime(t))
    return stime


def getNormalization(li):
    temp = []
    _max = max(li)
    _min = min(li)
    for i in li:
        normal = (i - _min) / (_max - _min)
        temp.append(normal)
    return temp


# 使用小波滤波
def cwt_filter(data, threshold):
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波

    data_filter = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
    if (len(data) != len(data_filter)):
        data_filter = np.delete(data_filter, 0)

    return data_filter


# 使用快速傅里叶变换滤波
def fft_filter(data, sampling_frequency, threshold_frequency):
    fft_result = np.fft.fft(data)
    begin = int(len(data) * threshold_frequency * sampling_frequency)
    fft_result[begin:] = 0  # 低通滤波
    filter_data = np.fft.ifft(fft_result)
    return abs(filter_data)


def draw_features_from_db(action, db, volt_collection, tag_collection, port=27017,
                           host='localhost', ndevices=3, offset=0, action_num=0,
                           interval=1, rate=1):
    client = MongoClient(port=port, host=host)
    database = client[db]
    tag_collection = database[tag_collection]
    volt_collection = database[volt_collection]

    try:
        if volt_collection.count_documents({}) + tag_collection.count_documents({}) < 2:
            raise CollectionError('Collection not found！')
    except CollectionError as e:
        print(e.message)

    # 定义特征提取器
    extractor = FeatureExtractor()

    for feature in feature_names:
        # 定义特征提取模块
        module = eval(feature + "(" + str(interval) + "," + str(rate) + ")")
        # 注册特征提取模块
        extractor.register(module)

    # ntags表示总标签数，即人数；tag_acc表示累加计数
    ntags = tag_collection.count_documents({'tag': action})
    # ntags = 1
    tag_acc = 0

    # 读取模型
    start = 1
    end = start
    device_no = 1
    import pickle
    with open('models/' + 'device_' + str(device_no) + '_post_prune.pickle', 'rb') as f:
        model = pickle.load(f)

    # read the data that is of a certain action one by one
    for tag in tag_collection.find({'tag': action}):
        tag_acc += 1
        if (tag_acc > ntags):
            break
        # if(tag_acc in discard[action]):
        # if (tag_acc == 9 or tag_acc == 11):
        if (tag_acc == 9 or (tag_acc == 11 and action != "hands_move")):
            continue
        inittime, termtime = tag['inittime'], tag['termtime']
        # 定义总特征计数
        feature_acc = 0
        # 定义准确率计数
        accuracy_acc = 0

        # get the arrays according to which we will plot later
        times, volts, filter_volts = {}, {}, {}
        for i in range(1, ndevices + 1):
            times[i] = []
            volts[i] = []
            filter_volts[i] = []

        for volt in volt_collection.find({'time': {'$gt': inittime, '$lt': termtime}}):
            device_no = int(volt['device_no'])
            v = volt['voltage']
            t = volt['time']
            times[device_no].append(t)
            volts[device_no].append(v)

        for i in range(start, end + 1):
            filter_volts[i] = volts[i]
            # 小波变换滤波
            filter_volts[i] = cwt_filter(volts[i], 0.08)

            # 低通滤波器滤波
            # b, a = signal.butter(8, 4 / 7, 'lowpass')  # 配置滤波器，8表示滤波器的阶数
            # filter_volts[i] = signal.filtfilt(b, a, filter_volts[i])

            # 傅里叶变换滤波，使用后动作识别准确率反而降低
            # filter_volts[i] = fft_filter(filter_volts[i], 1 / 70, 25)  #滤波后准确率下降

            # 除以体重，归一化数据
            # filter_volts[i] = list(map(lambda x: x / weights[tag_acc - 1], filter_volts[i]))
            filter_volts[i] = getNormalization(filter_volts[i])

        # 定义存储时间、特征列表
        feature_times, feature_values = {}, {}
        for i in range(start, end + 1):
            feature_times[i] = []
            from collections import defaultdict
            feature_values[i] = defaultdict(list)
            for feature in feature_names:
                feature_values[i][feature[:-6]] = []

        # 对每个采集设备进行特征提取 ndevices
        for i in range(start, end + 1):
            result = []
            for j in range(len(filter_volts[i])):
                value = {
                    "time": times[i][j],
                    "volt": filter_volts[i][j]
                }
                output = extractor.process(value)
                if (output):
                    feature_acc += 1
                    features = {
                        "device_no": i,
                        "feature_time": times[i][j],
                        "feature_value": output,
                        "interval": interval,
                        "rate": rate
                    }
                    feature_temp = []  #存储实时计算的一条特征数据
                    for feature_type in feature_values[i].keys():
                        # print(feature_type, features['feature_value'][feature_type])
                        feature_temp.append(features['feature_value'][feature_type])

                    print(feature_temp)

                    predict_result = model.predict([feature_temp])
                    result.append(predict_result)
                    format_result = "person" + str(tag_acc) + " " + "device_" + str(i)
                    # print(format_result, action_names[predict_result[0]])
                    if(predict_result[0]==action_num):
                        accuracy_acc += 1
            print(result)
            # 清理所有模块，防止过期数据
            extractor.clear()

        print("action:", action_names[action_num])
        print("person" + str(tag_acc) + "的准确率：", accuracy_acc/feature_acc)


if __name__ == '__main__':

    # for i in range(len(action_names)):
    i = 0
    draw_features_from_db(action=action_names[i],
                           db=config['db'],
                           tag_collection=config['tag_collection'],
                           volt_collection=config['volt_collection'],
                           ndevices=config['device_num'],
                           offset=config['offset'],
                           action_num=i,
                           interval=config['interval'],
                           rate=config['rate'])
