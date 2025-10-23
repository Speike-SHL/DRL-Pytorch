import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import copy
import os


def build_net(layer_shape: list, activation, output_activation):
    """
    快速构建全连接网络
    layer_shape: 一个表示每层神经元个数的列表
    activation: 隐藏层的激活函数
    output_activation: 输出层的激活函数
    """
    layers = []
    for j in range(len(layer_shape) - 1):
        act = activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Q_Net(nn.Module):
    """
    普通的 Q 网络
    nn.Linear(state_dim, hid_shape[0])
    nn.ReLU()
    nn.Linear(hid_shape[0], hid_shape[1])
    nn.ReLU()
    nn.Linear(hid_shape[1], action_dim)
    """

    def __init__(self, state_dim, action_dim, hid_shape):
        super(Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        # nn.Identity() 表示不做任何操作, 直接输出
        self.Q = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q = self.Q(s)
        return q


class Duel_Q_Net(nn.Module):
    """
    对决网络:
    共用部分: self.hidden
    nn.Linear(state_dim, hid_shape[0])
    nn.ReLU()
    nn.Linear(hid_shape[0], hid_shape[1])
    nn.ReLU()
    状态网络: self.V: nn.Linear(hid_shape[-1], 1)
    优势网络: self.A: nn.Linear(hid_shape[-1], action_dim)
    """

    def __init__(self, state_dim, action_dim, hid_shape):
        super(Duel_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape)
        self.hidden = build_net(layers, nn.ReLU, nn.ReLU)
        self.V = nn.Linear(hid_shape[-1], 1)
        self.A = nn.Linear(hid_shape[-1], action_dim)

    def forward(self, s):
        s = self.hidden(s)
        Adv = self.A(s)  # [batch_size, action_dim]
        V = self.V(s)  # [batch_size, 1]
        Q = V + (Adv - torch.mean(Adv, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q  # 输出为 [batch_size, action_dim]


class DQN_agent(object):
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005  # 目标网络软更新的比例
        self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))
        if self.Duel:
            self.q_net = Duel_Q_Net(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(self.dvc)
        else:
            self.q_net = Q_Net(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(self.dvc)
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        # 好像使用 self.q_target.load_state_dict(self.q_net.state_dict()) 更高效
        self.q_target = copy.deepcopy(self.q_net)
        # 完全冻结相对于优化器中的目标网络梯度（仅后面进行软更新）,
        # 这种相比于.eval, 可以完全避免计算梯度, 而eval某些情况还会计算, 只不过其中的dropout层等失效而已
        for p in self.q_target.parameters():
            p.requires_grad = False

    def select_action(self, state, deterministic):  # only used when interact with the env
        with torch.no_grad():
            # 转换维度, 因为输入网络正向传播时, 要求输入为 [batch_size, state_dim]
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
            # NOTE: 在测试时也可以选择最低概率的小探索, 用来测试策略是否稳健
            if deterministic:
                a = self.q_net(state).argmax().item()
            else:
                if np.random.rand() < self.exp_noise:
                    a = np.random.randint(0, self.action_dim)
                else:
                    a = self.q_net(state).argmax().item()
        return a

    def train(self):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        """计算下一状态的最大q值时, 不计算梯度"""
        with torch.no_grad():
            if self.Double:  # 如果使用 Double Q, 先用主网络选出最优动作, 再用目标网络计算q值
                # q_net->[batch_size, action_dim], argmax->[batch_size], unsqueeze->[batch_size, 1]
                argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
                # q_target->[batch_size, action_dim], gather->[batch_size, 1]
                max_q_next = self.q_target(s_next).gather(1, argmax_a)
            else:  # 如果不用 Double Q, 直接用目标网络计算最大q值
                # q_target->[batch_size, action_dim], max->[batch_size], unsqueeze->[batch_size, 1]
                # max(1) 在 action_dim 维度上取最大值, 范围值和对应索引, [0]是取值
                max_q_next = self.q_target(s_next).max(1)[0].unsqueeze(1)
            target_Q = r + (~dw) * self.gamma * max_q_next  # dw: die or win

        # Get current Q estimates
        current_q = self.q_net(s)  # [batch_size, action_dim], 所有动作的 q 值
        current_q_a = current_q.gather(1, a)  # [batch_size, 1], 实际选择的动作的 q 值

        # 计算 loss, 清空梯度, 反向传播, 更新参数
        # QUERY 没进行梯度截断
        q_loss = F.mse_loss(current_q_a, target_Q)
        self.q_net_optimizer.zero_grad()
        q_loss.backward()
        self.q_net_optimizer.step()

        # 软更新目标网络参数
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, algo, EnvName, steps):
        """
        :param steps: 训练的步数, 单位为千步
        """
        torch.save(self.q_net.state_dict(), "./2.1_Duel-Double-DQN/model/{}_{}_{}k.pth".format(algo, EnvName, steps))

    def load(self, algo, EnvName, steps):
        model_path = "./2.1_Duel-Double-DQN/model/{}_{}_{}k.pth".format(algo, EnvName, steps)
        if not os.path.exists(model_path):
            print(f"No model {model_path} found, please train first.")
            return False
        self.q_net.load_state_dict(torch.load(model_path, map_location=self.dvc))
        self.q_target.load_state_dict(torch.load(model_path, map_location=self.dvc))
        return True


class ReplayBuffer(object):
    def __init__(self, state_dim, dvc, max_size=int(1e6)):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0  # 下一个要存储的位置
        self.size = 0  # 当前缓冲区中存储的经验数量

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.a = torch.zeros((max_size, 1), dtype=torch.long, device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # 抽样可能重复
        # replayerbuffer 的目的主要是打破数据的相关性, 同样一条数据本来就会被使用多次
        # 因此当 buffer 大小远大于 batch_size 时, 抽样时允许重复采样
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
