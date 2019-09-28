from feature_extractor import FeatureExtractor
from feature_extractor import ProcessModule


class VibrationFreqModule(ProcessModule):
    """计算检测到的电压中的频率"""

    FEATURE_NAME = "VibrationFreq"

    def processFullQueue(self):
        volts = []
        for i in self.queue.queue:
            volts.append(i['volt'])
        import numpy as np
        # fft返回值实部表示
        result = np.fft.fft(volts)  # 除以长度表示归一化处理
        # fftfreq第一个参数n是FFT的点数，一般取FFT之后的数据的长度（size）
        # 第二个参数d是采样周期，其倒数就是采样频率Fs，即d=1/Fs
        freq = np.fft.fftfreq(len(result), d=1 / 70)
        amplitude = np.sqrt(result.real ** 2 + result.imag ** 2) / (len(volts) / 2)
        sum = 0
        for f, a in zip(freq, amplitude):
            sum += abs(f) * a
        return sum / len(freq)

    def get_average(self, list):
        sum = 0
        for item in list:
            sum += item
        return sum / len(list)

    def clear(self):
        """清理组件中的队列"""
        self.queue.queue.clear()


class EnergyModule(ProcessModule):
    """计算检测到的电压中的能量：指定时间间隔内不同频率对应的幅值之和"""

    FEATURE_NAME = "Energy"

    def processFullQueue(self):
        volts = []
        for i in self.queue.queue:
            volts.append(i['volt'])
        import numpy as np
        # fft返回值实部表示
        result = np.fft.fft(volts)  # 除以长度表示归一化处理
        # fftfreq第一个参数n是FFT的点数，一般取FFT之后的数据的长度（size）
        # 第二个参数d是采样周期，其倒数就是采样频率Fs，即d=1/Fs
        freq = np.fft.fftfreq(len(result), d=1 / 70)
        amplitude = np.sqrt(result.real ** 2 + result.imag ** 2) / (len(volts) / 2)
        sum = 0
        for i in abs(amplitude):
            sum += i
        return sum

    def get_average(self, list):
        sum = 0
        for item in list:
            sum += item
        return sum / len(list)

    def clear(self):
        """清理组件中的队列"""
        self.queue.queue.clear()


class RangeModule(ProcessModule):
    """计算检测到的电压中的极差"""

    FEATURE_NAME = "Range"

    def processFullQueue(self):
        range = []
        for i in self.queue.queue:
            range.append(i['volt'])
        return max(range) - min(range)

    def clear(self):
        """清理组件中的队列"""
        self.queue.queue.clear()


class DurationModule(ProcessModule):
    """计算从最大值到最小值所花的时间"""

    FEATURE_NAME = "Duration"

    def processFullQueue(self):
        import sys
        max = 0
        min = sys.maxsize
        for i in self.queue.queue:
            if (i['volt'] > max):
                max = i['volt']
                max_time = i['time']
            if (i['volt'] < min):
                min = i['volt']
                min_time = i['time']
        return 2 - abs(max_time - min_time)

    def clear(self):
        """清理组件中的队列"""
        self.queue.queue.clear()


class SamplingFreqModule(ProcessModule):
    """计算设备指定采样时间间隔内采样的数量"""

    FEATURE_NAME = "SamplingFreq"

    def processFullQueue(self):
        # 使用self.size得到了不一样的容量大小
        return len(self.queue.queue) / self.interval

    def clear(self):
        """清理组件中的队列"""
        self.queue.queue.clear()


class ThresholdCounterModule(ProcessModule):
    """求超过指定阈值的个数"""

    FEATURE_NAME = "ThresholdCounter"
    UPPER_THRESHOLD = 0.80
    LOWER_THRESHOLD = 0.80

    def processFullQueue(self):
        count = 0
        sum = 0
        # for value in self.queue.queue:
        #     sum = sum + value['volt']
        # average = sum / self.size
        # for value in self.queue.queue:
        #     if value['volt'] > average:
        #         count += 1
        for value in self.queue.queue:
            if value['volt'] >= self.UPPER_THRESHOLD or value['volt'] <= self.LOWER_THRESHOLD:
                count += 1
        return count

    def clear(self):
        """清理组件中的队列"""
        self.queue.queue.clear()


class AverageModule(ProcessModule):
    """功能是对满队列中的所有数据求平均值。返回平均值。
    表示震动幅度的平均情况"""
    # 优化，不用重复计算值，只计算增加和减少的数据；可以从其它组件获取数据

    FEATURE_NAME = "Average"

    def processFullQueue(self):
        sum = 0
        for value in self.queue.queue:
            sum = sum + value['volt']
        return sum / self.size

    def clear(self):
        """清理组件中的队列"""
        self.queue.queue.clear()


class VarianceModule(ProcessModule):
    """功能是对满队列中的所有数据求方差。返回方差。
    表示震动频率的平均情况"""

    FEATURE_NAME = "Variance"

    def processFullQueue(self):
        sum = 0
        variance = 0
        for value in self.queue.queue:
            sum = sum + value['volt']
        average = sum / self.size
        for value in self.queue.queue:
            variance = variance + (value['volt'] - average) ** 2
        return variance / self.size

    def clear(self):
        """清理组件中的队列"""
        self.queue.queue.clear()
