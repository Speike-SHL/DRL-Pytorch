import torch.nn.functional as F
import torch.nn as nn
import torch


class Actor(nn.Module):
    """Actor 网络
    1. 输入为状态, 输出为动作
    2. 三层全连接网络构造的多层感知机
    3. 前两层都使用 tanh 激活函数（因为需要控制范围, 使后面softmax函数的输出更符合概率分布。
        同时平滑的输出利于策略更新。此外，tanh能够更充分的对所有神经元激活，利于保持策略的探索性）
    4. 最后一层使用 softmax 激活函数, 输出为动作概率分布。
    """

    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim=0):
        """pi 函数为策略, 输出每个动作的概率分布
        softmax_dim 默认为0, 原因是动作维度为1, 所以默认对第一个维度进行softmax
        """
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob


class Critic(nn.Module):
    """Critic 网络
    1. 输入为状态, 输出为状态值
    2. 三层全连接网络构造的多层感知机
    3. 前两层都使用 relu 激活函数, 最后一层使用 linear 激活函数
    4. 输出为状态值, 单值, 表示状态值的大小
    """

    def __init__(self, state_dim, net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.relu(self.C1(state))
        v = torch.relu(self.C2(v))
        v = self.C3(v)
        return v


def evaluate_policy(env, agent, turns=3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset(seed=j * 2026)  # 评估时固定随机种子, 确保有相同的评估基准
        done = False
        while not done:
            # Take deterministic actions at test time
            a, logprob_a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = dw or tr

            total_scores += r
            s = s_next
    return int(total_scores / turns)


# You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    """transfer str to bool for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "True", "true", "TRUE", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "False", "false", "FALSE", "f", "n", "0"):
        return False
    else:
        print("Wrong Input.")
        raise
