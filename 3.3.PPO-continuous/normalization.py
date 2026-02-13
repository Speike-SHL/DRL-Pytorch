import numpy as np


class RunningMeanStd:
    # 动态计算均值和方差, 由前n-1个数据的均值和方差以及第n个数据计算出前n个数据的均值和方差
    # 得到一个均值为0，方差为1的标准化数据
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)  # 当前累积折扣奖励，即回报

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        # 计算累积折扣奖励的均值和方差
        self.running_ms.update(self.R)
        # 只缩放，不平移，保持奖励的相对大小关系不变，因为奖励的相对大小包含了重要的信息
        # 用折扣奖励标准化而不是用即时奖励标准化，因为累积折扣更反应状态价值,更稳定
        # QUERY: 这样移动缩放会不会不好, 是不是应该一个batch后按整个回合每一步的累积回报算出均值和方差
        # QUERY: 然后再统一缩放每一步的奖励?
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)
