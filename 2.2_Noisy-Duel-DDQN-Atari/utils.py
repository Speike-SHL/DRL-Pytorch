import torch.nn.functional as F
from torch import nn
import torch
import math


def evaluate_policy(env, agent, seed, turns=3):
    agent.q_net.eval()  # 关闭NoisyNet的噪声
    scores = 0
    for j in range(turns):
        s, info = env.reset(seed=seed)
        done = False
        while not done:
            a = agent.select_action(s, evaluate=True)  # choose action with e-greedy = 0.01
            s_next, r, dw, tr, info = env.step(a)  # dw(dead & win): terminated, tr: truncated
            done = dw or tr
            scores += r
            s = s_next
    agent.q_net.train()
    return int(scores / turns)


# You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    """Fix the bool BUG for argparse: transfer string to bool"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "True", "true", "TRUE", "t", "y", "1", "T"):
        return True
    elif v.lower() in ("no", "False", "false", "FALSE", "f", "n", "0", "F"):
        return False
    else:
        print("Wrong Input Type!")


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """在 schedule_timesteps 步数内，将 initial_p 线性插值到 final_p。
        超过该步数后，始终返回 final_p。

        参数
        ----------
        schedule_timesteps: int
            线性退火的总步数，用于将 initial_p 逐步降低到 final_p
        initial_p: float
            初始输出值
        final_p: float
            最终输出值
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """
        Args:
            t (_type_): 当前步数
        Returns:
            _type_: 当前 e-greedy noise
        """
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class NoisyLinear(nn.Module):
    """From https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/3.Rainbow_DQN/network.py
    创建一个噪声线性层代替原来的全连接层
    w = w_mu + w_sigma * ksi, w_mu 和 w_sigma 都是可学习的参数, ksi 是从标准正态分布中采样的噪声,
    相当于 w 是从 w_mu 为均值, w_sigma 为标准差的正态分布中采样的.
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # 先 out_features, 再 in_features, 原因是 pytorch 的 Linear 计算是 y = x @ w.T + b, 即 w 是 out_features x in_features
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # register_buffer 是模型的一个函数, 用于注册那些不参与模型更新, 但是需要跟随模型一起保存和加载的张量
        # 这里 epsilon(即ksi) 是从标准正态分布中采样的噪声
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()  # for mu and sigma
        self.reset_noise()  # for epsilon

    def forward(self, x):
        if self.training:
            # NOTE: 训练时每回合都重置噪声
            self.reset_noise()
            # (out, in) + (out, in) * (out, in) = (out, in)
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)  # mul是对应元素相乘
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            # NOTE: 评估时不使用噪声
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)  # 最后输出 y = x @ w.T + b

    def reset_parameters(self):
        """
        将 mu 参数使用 uniform 均匀采样, 负责初始化时的探索
        将 sigma 参数用 fill 去全部填充一样的固定值, 稳定初始化时的不确定性
        至于 sqrt(in_features), 是为了控制网络的标准差, 使训练更稳定. 因为模型的标准差会随着神经元数量增大减小
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        # epsilon_j 和 epsilon_i 做外积, 得到一个 out_features x in_features 的噪声矩阵
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        """
        先生成标准正态分布的随机噪声, 然后用每个变量的符号和该变量绝对值的平方根相乘
        效果上就是可以缩放一些很大的异常值, 因为偏离很多的值开方后缩小了。
        """
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
