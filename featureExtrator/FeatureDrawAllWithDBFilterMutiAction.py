from feature_modules import *
from feature_extractor import FeatureExtractor
import time

from pymongo import MongoClient
from exceptions import CollectionError
from pymongo import MongoClient

from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np

action = ["still", "turn_over", "legs_stretch", "hands_stretch",
          "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]

config = {'action': action[1],
          'db': 'beaglebone',
          'tag_collection': 'tags_424',
          'volt_collection': 'volts_424',
          'offset': 0}

# 导入全局变量
import GlobalVariable as gv
action = gv.action_names
# feature_names = gv.feature_names

config = {'action': action,
          'db': 'beaglebone',
          'tag_collection': 'tags_1105',
          'volt_collection': 'volts_1105',
          'ndevices': 5,
          'offset': 0
          }

ndevices = 5
start = 5
end = start


# feature_names = ["MeanModule", "SDModule"]
# feature_names = ["MeanModule"]
feature_names = ["RangeModule", "MeanModule", "SDModule", "EnergyModule", "FDEModule",
                 "RMSModule"]


def timeToFormat(t):
    ftime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    return ftime


def timeToSecond(t):
    stime = time.strftime("%M:%S", time.localtime(t))
    return stime


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


def getNormalization(li):
    temp = []
    _max = max(li)
    _min = min(li)
    for i in li:
        normal = (i - _min) / (_max - _min)
        temp.append(normal)
    return temp


def draw_features_from_db(action, db, volt_collection, tag_collection, port=27017, host='localhost',
                          ndevices=3, offset=0, action_num=0, feature_name="Mean"):
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
    ntags = 3
    tag_acc = 0

    # 根据时间采集数据，基本单位为s，比如1s、10s、30s、60s
    # interval表示每次分析的时间跨度，rate表示间隔多长时间进行一次分析
    interval = 2
    rate = 1

    # 定义特征提取器
    extractor = FeatureExtractor()

    for feature in feature_names:
        # 定义特征提取模块
        module = eval(feature + "(" + str(interval) + "," + str(rate) + ")")
        # 注册特征提取模块
        extractor.register(module)

    # read the data that is of a certain action one by one
    for tag in tag_collection.find({'tag': action}):
        tag_acc += 1
        if(tag_acc < ntags):
            continue
        if(tag_acc > ntags):
            break
        # inittime
        # inittime, termtime = tag['inittime'], tag['termtime']
        inittime, termtime = tag['termtime'] - 31, tag['termtime']

        # get the arrays according to which we will plot later
        times, volts, filter_volts = {}, {}, {}
        for i in range(1, ndevices + 1):
            times[i] = []
            volts[i] = []
            filter_volts[i] = []

        for volt in volt_collection.find({'time': {'$gt': inittime, '$lt': termtime}}):
            device_no = int(volt['device_no'])
            v = volt['voltage']
            time = volt['time']
            times[device_no].append(time)
            volts[device_no].append(v)

        # 滤波
        for i in range(start, end + 1):
            # 小波变换滤波
            filter_volts[i] = cwt_filter(volts[i], 0.08)

            # 傅里叶变换滤波
            # filter_volts[i] = fft_filter(filter_volts[i], 1 / 70, 25)

            # 归一化数据
            filter_volts[i] = getNormalization(filter_volts[i])

        # 定义存储时间、特征列表
        # feature_times, feature_values = {}, {}
        # for i in range(start, end + 1):
        #     feature_times[i] = []
        #     from collections import defaultdict
        #     feature_values[i] = defaultdict(list)
        #     for feature in feature_names:
        #         feature_values[i][feature[:-6]] = []

        # 对每个采集设备进行特征提取 ndevices
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
                    fea_diff_action[action_num].append(features['feature_value'][feature_name])

            # 清理所有模块，防止过期数据
            extractor.clear()


if __name__ == '__main__':
    for j in range(len(feature_names)):
        fea_diff_action = []
        for i in range(len(action)):
            fea_diff_action.append([])

        for i in range(len(action)):
            print("---------" + action[i] + "---------")
            draw_features_from_db(action=action[i],
                         db=config['db'],
                         tag_collection=config['tag_collection'],
                         volt_collection=config['volt_collection'],
                         ndevices=5,
                         offset=config['offset'],
                         action_num=i,
                         feature_name=feature_names[j][:-6])
        title = config['volt_collection'][6:] + feature_names[j][:-6] + "_different_movement_device" + str(start)
        plt.figure(title)
        plot_colors = ['r', 'm', 'c', 'b', 'g', 'lime', 'y', 'peru', 'navy', 'orange']
        # fea_x_axle = range(1, 30) #有些设备丢失数据，需要对齐处理
        for i in range(len(action)):
            plt.plot(fea_diff_action[i], color=plot_colors[i], label=action[i])
        plt.xlabel('Time(s)')
        plt.ylabel(feature_names[j][:-6]+"(mv)")
        plt.title('Feature values of different sleep movements of device' + str(start))
        plt.legend(loc="upper right")
        # plt.get_current_fig_manager().window.state('zoomed')
        # plt.show()
        plt.savefig("feature_images/" + title + ".png")
        plt.close()