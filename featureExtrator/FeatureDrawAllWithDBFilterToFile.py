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

config = {'action': action,
          'db': 'beaglebone',
          'tag_collection': 'tags_1105',
          'volt_collection': 'volts_1105',
          'ndevices': 5,
          'offset': 0
          }

ndevices = 5
start = 1
end = ndevices

# feature_names = ["RangeModule", "EnergyModule", "RMSModule", "FDEModule","SamplingFreqModule"]
# feature_names = ["SamplingFreqModule", "ThresholdCounterModule"]
# feature_names = ["MeanModule", "SDModule"]
feature_names = ["RangeModule", "MeanModule", "SDModule", "EnergyModule", "FDEModule", "RMSModule"]


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


def draw_features_from_db(action, db, volt_collection, tag_collection, port=27017, host='localhost', ndevices=3, offset=0):
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
    ntags = 1
    tag_acc = 0

    title = config['volt_collection'][6:] + "" + action + "_features"
    # fig = plt.figure(title, figsize=(6, 8))
    fig = plt.figure(title)

    # 根据时间采集数据，基本单位为s，比如1s、10s、30s、60s
    # interval表示每次分析的时间跨度，rate表示间隔多长时间进行一次分析
    interval = 2
    rate = 1
    fig.suptitle(action + "_features (" + "interval:" + str(interval) + "s, " + "stepsize:" + str(rate) + "s)")

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
        if(tag_acc > ntags):
            break
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
        feature_times, feature_values = {}, {}
        for i in range(start, end + 1):
            feature_times[i] = []
            from collections import defaultdict
            feature_values[i] = defaultdict(list)
            for feature in feature_names:
                feature_values[i][feature[:-6]] = []

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
                    feature_times[i].append(features['feature_time'])
                    for feature_type in feature_values[i].keys():
                        feature_values[i][feature_type].append(features['feature_value'][feature_type])

            # 清理所有模块，防止过期数据
            extractor.clear()

        # 定义特征数量
        nfeatures = len(feature_values[1])
        # 定义画布上下位置的计数，即特征累加
        fea_acc = 0

        style.use('default')
        colors = ['r', 'b', 'g', 'c', 'm']  # m c

        for feature_type in feature_values[1].keys():
            fea_acc += 1
            ax = fig.add_subplot(nfeatures, ntags, (fea_acc-1) * ntags + tag_acc)
            plt.subplots_adjust(hspace=0.5)  # 函数中的wspace是子图之间的垂直间距，hspace是子图的上下间距
            ax.set_title(feature_type)

            for i in range(start, end + 1):
                ax.set_xlim(feature_times[i][0], feature_times[i][-1])
                ax.plot(feature_times[i], feature_values[i][feature_type],
                        label='device_' + str(i), color=colors[i - 1], alpha=0.9)

                # # 获取最大最小值，并且打上标记
                # max_index = np.argmax(feature_values[i][feature_type])
                # min_index = np.argmin(feature_values[i][feature_type])
                # ax.plot(feature_times[i][max_index],feature_values[i][feature_type][max_index],'rs')
                # show_max = str(i)+":"+str(round(feature_values[i][feature_type][max_index],6))
                # # xy=(横坐标，纵坐标)  箭头尖端, xytext=(横坐标，纵坐标) 文字的坐标，指的是最左边的坐标
                # # https://blog.csdn.net/qq_30638831/article/details/79938967
                # plt.annotate(show_max, xy=(feature_times[i][max_index],
                #     feature_values[i][feature_type][max_index]),
                #     xytext=(feature_times[i][max_index], feature_values[i][feature_type][max_index]))
                # ax.plot(feature_times[i][min_index], feature_values[i][feature_type][min_index], 'gs')
                # show_min = str(i)+":"+str(round(feature_values[i][feature_type][min_index],6))
                # plt.annotate(show_min, xy=(feature_times[i][min_index],
                #     feature_values[i][feature_type][min_index]),
                #     xytext=(feature_times[i][min_index], feature_values[i][feature_type][min_index]))

            # 设置每个数据对应的图像名称
            if fea_acc == 1 and tag_acc == 1:
                ax.legend(loc='upper right')
                ax.set_xlabel('Time(s)')
            if fea_acc == nfeatures:
                # 设置人员
                person = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
                ax.set_xlabel("Person" + person[tag_acc - 1] + ": " + timeToFormat(inittime + offset)
                              + " ~ " + timeToFormat(termtime + offset))

            # 以第一个设备的时间数据为准，数据的每1/10添加一个x轴标签
            xticks = []
            xticklabels = []
            length = len(feature_times[i])
            step = 2
            for k in range(0, length, step):
                xticks.append(feature_times[i][k])
                # xticklabels.append(timeToSecond(feature_times[i][k] + offset))
                xticklabels.append(int(feature_times[i][k] - inittime))  # 图中的开始时间表示时间间隔interval
            # 设定标签的实际数字，数据类型必须和原数据一致
            ax.set_xticks(xticks)
            # 设定我们希望它显示的结果，xticks和xticklabels的元素一一对应
            ax.set_xticklabels(xticklabels, rotation=15)

            # 显示网格
            ax.grid(linestyle=':')
            # ax.grid(True, which='both')

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(20, 10)
    plt.savefig("feature_images/" + title + ".png", dpi=200)
    plt.close()

    # 最大化显示图像窗口
    # plt.get_current_fig_manager().window.state('zoomed')
    # plt.show()


if __name__ == '__main__':
    for i in range(len(action)):
        print("---------" + action[i] + "---------")
        draw_features_from_db(action=action[i],
                     db=config['db'],
                     tag_collection=config['tag_collection'],
                     volt_collection=config['volt_collection'],
                     ndevices=5,
                     offset=config['offset'])