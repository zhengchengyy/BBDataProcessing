from pymongo import MongoClient
from matplotlib import pyplot as plt
from matplotlib import style
from exceptions import CollectionError
from scipy import signal
import numpy as np
import time
import pywt

action = ["still", "turn_over", "legs_stretch", "hands_stretch",
              "legs_twitch", "hands_twitch", "head_move", "grasp", "kick"]

config = {'action': action[1],
          'db': 'beaglebone',
          'tag_collection': 'tags_411',
          'volt_collection': 'volts_411',
          'offset': 0}


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

def fft_filter(data, sampling_frequency, threshold_frequency):
    fft_result = np.fft.fft(data)
    begin = int(len(data) * threshold_frequency * sampling_frequency)
    fft_result[begin:] = 0  # 高通滤波
    filter_data = np.fft.ifft(fft_result)
    return abs(filter_data)

def plot_from_db(action, db, volt_collection, tag_collection, port=27017, host='localhost',
                 ndevices=3, offset=0):
    client = MongoClient(port=port, host=host)
    database = client[db]
    tag_collection = database[tag_collection]
    volt_collection = database[volt_collection]

    try:
        if volt_collection.count_documents({}) + tag_collection.count_documents({}) < 2:
            raise CollectionError('Collection not found!')
    except CollectionError as e:
        print(e.message)

    # ntags表示总标签数，即人数；tag_acc表示累加计数
    ntags = tag_collection.count_documents({'tag': action})
    # ntags = 8
    tag_acc = 0

    # 用于查看几号设备的图
    start = 3
    end = ndevices

    title = config['volt_collection'][6:] + "" + action + "_fft_filter_" + str(start)
    fig = plt.figure(title, figsize=(6, 8))
    fig.suptitle(action + "_fft_filter")

    # plot the data that is of a certain action one by one
    for tag in tag_collection.find({'tag': action}):
        tag_acc += 1
        if (tag_acc > ntags):
            break
        # inittime
        inittime, termtime = tag['inittime'] - offset, tag['termtime'] - offset
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

        style.use('default')
        colors = ['r', 'b', 'g', 'c', 'm']
        subtitle = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H','I','J','L','M','N']
        ax = fig.add_subplot(ntags, 1, tag_acc)
        plt.subplots_adjust(hspace=0.5)  # 函数中的wspace是子图之间的垂直间距，hspace是子图的上下间距
        ax.set_title("Person" + subtitle[tag_acc - 1] + ": " + timeToFormat(inittime + offset)
                     + " ~ " + timeToFormat(termtime + offset))

        # 自定义y轴的区间范围，可以使图放大或者缩小
        # ax.set_ylim(0, 0.001)
        # ax.set_ylim(0, 0.0003)
        # ax.set_ylim(0, 1)
        ax.set_ylabel('Amplitude')

        filter_thread = [0.2, 0.06, 0.08]
        for i in range(start, end + 1):
            ax.plot(times[i], volts[i], label='device_' + str(i), color=colors[i - 1], alpha=0.3)
            # 小波变换滤波
            filter_volts[i] = cwt_filter(volts[i], filter_thread[i - 1])

            # 傅里叶变换滤波
            filter_volts[i] = fft_filter(filter_volts[i], 1 / 70, 15)

            ax.plot(times[i], filter_volts[i], label='device_' + str(i) + "_cwt_filter", color=colors[i - 1], alpha=0.9)

            # 低通滤波器滤波，滤除15Hz以上的频率成分，wn = 2 * 15 / 70
            # b, a = signal.butter(8, 3 / 7, 'lowpass')  # 配置滤波器，8表示滤波器的阶数
            # b, a = signal.butter(8, [1 / 7, 4 / 7], 'bandpass')
            # filter_volts[i] = signal.filtfilt(b, a, filter_volts[i])

            # fft返回值实部表示
            # result = np.fft.fft(filter_volts[i])
            # # 实数fft后会使原信号幅值翻N/2倍，直流分量即第一点翻N倍
            # amplitudes = abs(result) / (len(result) / 2)  # 复数的绝对值其实就是它的模长
            # amplitudes[0] /= 2
            # # fftfreq第一个参数n是FFT的点数，一般取FFT之后的数据的长度（size）
            # # 第二个参数d是采样周期，其倒数就是采样频率Fs，即d=1/Fs
            # freqs = np.fft.fftfreq(len(result), d=1 / 70)
            # ax.plot(abs(freqs), amplitudes, label='device_' + str(i), color=colors[i - 1], alpha=0.9)

        if tag_acc == 1:
            ax.legend(loc='upper right')
        if tag_acc == ntags:
            ax.set_xlabel('Frequency')

    # 最大化显示图像窗口
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


if __name__ == '__main__':
    plot_from_db(action=config['action'],
                 db=config['db'],
                 tag_collection=config['tag_collection'],
                 volt_collection=config['volt_collection'],
                 offset=config['offset'])
