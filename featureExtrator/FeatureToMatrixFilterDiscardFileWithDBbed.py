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

from scipy import signal

config = {'db': 'beaglebone',
          'tag_collection': 'tags_1105',
          'volt_collection': 'volts_1105',
          'device_num': 5,
          'offset': 0,
          'interval': 2,
          'rate': 1}

# weights = [65, 75, 65, 45, 60, 62, 48, 55, 65, 60, 70, 65]

# 导入全局变量
import GlobalVariable as gv

# action_names = gv.action_names
action_names = ["get_up","go_to_bed"]
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


# 使用np实现移动平均滤波
def np_move_avg(a, n, mode="same"):
    return (np.convolve(a, np.ones((n,)) / n, mode=mode))


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
    freqs = np.fft.fftfreq(len(fft_result), d=sampling_frequency)
    begin = int(len(data) * threshold_frequency * sampling_frequency)
    fft_result[begin:] = 0  # 高通滤波
    filter_data = np.fft.ifft(fft_result)
    return abs(filter_data)


def feature_to_matrix_file(action, db, volt_collection, tag_collection, port=27017,
                          host='localhost', ndevices=3, offset=0, action_num=0,
                          interval = 1, rate = 1):
    # 根据时间采集数据，基本单位为s，比如1s、10s、30s、60s
    # interval表示每次分析的时间跨度，rate表示间隔多长时间进行一次分析
    # print(interval,rate)
    client = MongoClient(port=port, host=host)
    database = client[db]
    tag_collection = database[tag_collection]
    volt_collection = database[volt_collection]

    try:
        if volt_collection.count_documents({}) + tag_collection.count_documents({}) < 2:
            raise CollectionError('Collection not found！')
    except CollectionError as e:
        print(e.message)

    # ntags表示总标签数，即人数；tag_acc表示累加计数
    ntags = tag_collection.count_documents({'tag': action})
    tag_acc = 0

    title = config['volt_collection'][6:] + "" + action + "_features"
    fig = plt.figure(title, figsize=(6, 8))
    fig.suptitle(action + " (" + "interval:" + str(interval) + "s, " + "stepsize:" + str(rate) + "s)")

    # 定义特征提取器
    extractor = FeatureExtractor()

    for feature in feature_names:
        # 定义特征提取模块
        module = eval(feature + "(" + str(interval) + "," + str(rate) + ")")
        # 注册特征提取模块
        extractor.register(module)

    # 丢弃数据
    discard = {"get_up":[],"go_to_bed":[],
               "turn_over":[2,4,5,6,9],"legs_stretch":[2,5,7,8,9,11],
               "hands_stretch":[7,8,9],"legs_tremble":[3,6,9,10,11,12],
               "hands_tremble":[2,3,7,8,9,11],"body_tremble":[1,2,3,6,7,8,9],
               "head_move":[3,9,12],"legs_move":[1,3,4,6,9],"hands_move":[3,6,9],
               "hands_rising":[4,5,7,8,9],"kick":[2,4,5,6,7,8,9,11,12]
               }

    # read the data that is of a certain action one by one
    for tag in tag_collection.find({'tag': action}):
        tag_acc += 1
        if tag_acc in discard[action]:
        # if tag_acc == 9 or tag_acc == 11:  #don't discard data
            continue
        print("people_" + str(tag_acc))
        inittime, termtime = tag['inittime'], tag['termtime']
        # inittime, termtime = tag['inittime'] + 1, tag['termtime'] - 0.5

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

        for i in range(1, ndevices + 1):
            filter_volts[i] = volts[i]
            # 小波变换滤波
            filter_volts[i] = cwt_filter(volts[i], 0.08)

            # 低通滤波器滤波
            # b, a = signal.butter(8, 3 / 7, 'lowpass')  # 配置滤波器，8表示滤波器的阶数
            # filter_volts[i] = signal.filtfilt(b, a, filter_volts[i])

            # 傅里叶变换滤波
            filter_volts[i] = fft_filter(filter_volts[i], 1 / 70, 25)

            # 移动平均滤波，参数可选：full, valid, same
            # filter_volts[i] = np_move_avg(filter_volts[i], 5, mode="same")

            # 除以体重，归一化数据
            # filter_volts[i] = list(map(lambda x: x / weights[tag_acc - 1], filter_volts[i]))
            filter_volts[i] = getNormalization(filter_volts[i])

        # 定义存储时间、特征列表
        feature_times, feature_values = {}, {}
        for i in range(1, ndevices + 1):
            feature_times[i] = []
            from collections import defaultdict
            feature_values[i] = defaultdict(list)
            for feature in feature_names:
                feature_values[i][feature[:-6]] = []

        # 提取第几个设备的特征
        start = 1
        end = ndevices

        # 对每个采集设备进行特征提取
        for i in range(start, end + 1):
            for j in range(len(filter_volts[i])):
                value = {
                    "time": times[i][j],
                    "volt": filter_volts[i][j]
                }
                output = extractor.process(value)
                if (output):
                    features = {
                        "device_no": i,
                        "feature_time": times[i][j],
                        "feature_value": output,
                        "interval": interval,
                        "rate": rate
                    }
                    feature_times[i].append(features['feature_time'])
                    for feature_type in feature_values[i].keys():
                        feature_values[i][feature_type].append(features['feature_value'][feature_type])

            # 清理所有模块，防止过期数据
            extractor.clear()

        # 定义特征数量
        nfeatures = len(feature_values[1])

        # 定义特征类型
        feature_type = list(feature_values[1].keys())  # keys()方法虽然返回的是列表，但是不可以索引

        for i in range(start, end + 1):

            # 如果文件存在，则以添加的方式打开
            if (os.path.exists("feature_matrixs/feature_matrix_bed" + str(i) + ".npy")):
                feature_matrix = np.load("feature_matrixs/feature_matrix_bed" + str(i) + ".npy")
                label_matrix = np.load("feature_matrixs/label_matrix_bed" + str(i) + ".npy")
                temp_matrix = np.zeros((len(feature_times[i]), nfeatures), dtype=float)

                os.remove("feature_matrixs/feature_matrix_bed" + str(i) + ".npy")
                os.remove("feature_matrixs/label_matrix_bed" + str(i) + ".npy")

                for j in range(len(feature_times[i])):
                    for k in range(nfeatures):
                        temp_matrix[j][k] = feature_values[i][feature_type[k]][j]
                    label_matrix = np.append(label_matrix, [action_num])

                # np.append(feature_matrixs, [temp_matrix], axis=0)
                feature_matrix = np.insert(feature_matrix, feature_matrix.shape[0],
                                           values=temp_matrix, axis=0)

                np.save('feature_matrixs/feature_matrix_bed' + str(i), feature_matrix)
                np.save('feature_matrixs/label_matrix_bed' + str(i), label_matrix)

                print("feature_matrix_bed" + str(i) + ":" + str(feature_matrix.shape))


            # 如果文件不存在，则定义特征矩阵和标签矩阵
            else:
                feature_matrix = np.zeros((len(feature_times[i]), nfeatures), dtype=float)
                label_matrix = np.zeros((len(feature_times[i]), 1), dtype=int)

                for j in range(len(feature_times[i])):
                    for k in range(nfeatures):
                        feature_matrix[j][k] = feature_values[i][feature_type[k]][j]
                    label_matrix[j] = action_num
                # np.save保存时自动为8位小数
                np.save('feature_matrixs/feature_matrix_bed' + str(i), feature_matrix)
                np.save('feature_matrixs/label_matrix_bed' + str(i), label_matrix)

                print("feature_matrix_bed" + str(i) + ":" + str(feature_matrix.shape))


if __name__ == '__main__':
    # 清除文件
    start = 1
    end = config['device_num']
    for i in range(start, 5 + 1):
        if (os.path.exists("feature_matrixs/feature_matrix_bed" + str(i) + ".npy")):
            os.remove("feature_matrixs/feature_matrix_bed" + str(i) + ".npy")
            os.remove("feature_matrixs/label_matrix_bed" + str(i) + ".npy")

    for i in range(len(action_names)):
        print("---------" + action_names[i] + "---------")
        feature_to_matrix_file(action=action_names[i],
                              db=config['db'],
                              tag_collection=config['tag_collection'],
                              volt_collection=config['volt_collection'],
                              ndevices=config['device_num'],
                              offset=config['offset'],
                              action_num=i + 9,
                              interval=config['interval'],
                              rate=config['rate'])
