import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import NoisyLinear
import os


class Q_Net(nn.Module):
    def __init__(self, opt):
        super(Q_Net, self).__init__()
        # input, 4 channels, 84x84
        self.conv = nn.Sequential(
            # input 4 通道, output 32 通道, kernel 8x8, stride 4. out: 32x20x20
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            # input 32 通道, output 64 通道, kernel 4x4, stride 2. out: 64x9x9
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            # input 64 通道, output 64 通道, kernel 3x3, stride 1. out: 64x7x7
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            # 展平成全连接层输入. out: 64*7*7=3136
            nn.Flatten(),
        )
        if opt.Noisy:
            self.fc1 = NoisyLinear(64 * 7 * 7, opt.fc_width)
            self.fc2 = NoisyLinear(opt.fc_width, opt.action_dim)
        else:
            # in: 3136, out: fc_width
            self.fc1 = nn.Linear(64 * 7 * 7, opt.fc_width)
            # in: fc_width, out: action_dim
            self.fc2 = nn.Linear(opt.fc_width, opt.action_dim)

    def forward(self, obs):
        # 将像素值从 0-255 归一化到 0-1 的浮点数
        s = obs.float() / 255
        s = self.conv(s)
        s = torch.relu(self.fc1(s))
        q = self.fc2(s)
        return q


class Duel_Q_Net(nn.Module):
    def __init__(self, opt):
        super(Duel_Q_Net, self).__init__()
        # 这部分见 Q_Net 类中解析
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 主要还是, 在前面共用编码器(这里是卷积)后, 后面部分分为状态网络和优势网络分别输出
        if opt.Noisy:
            self.fc = NoisyLinear(64 * 7 * 7, opt.fc_width)
            self.A = NoisyLinear(opt.fc_width, opt.action_dim)
            self.V = NoisyLinear(opt.fc_width, 1)
        else:
            self.fc = nn.Linear(64 * 7 * 7, opt.fc_width)
            self.A = nn.Linear(opt.fc_width, opt.action_dim)
            self.V = nn.Linear(opt.fc_width, 1)

    def forward(self, obs):
        """解释见 2.1_Duel-Double-DQN"""
        s = obs.float() / 255  # convert to f32 and normalize before feeding to network
        s = self.conv(s)
        s = torch.relu(self.fc(s))
        Adv = self.A(s)
        V = self.V(s)
        Q = V + (Adv - torch.mean(Adv, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q


class DeepQ_Agent(object):
    def __init__(self, opt):
        self.dvc = opt.dvc
        self.action_dim = opt.action_dim
        self.batch_size = opt.batch_size
        self.gamma = opt.gamma
        self.train_counter = 0
        self.huber_loss = opt.huber_loss
        self.Double = opt.Double
        self.Duel = opt.Duel
        self.Noisy = opt.Noisy

        # 是否使用 Double 在 train 时判断, 看 TD Target 怎么算
        # 而 Noisy 主要是一种方法, 使用于多种网络, 因此可以在 Duel_Q_Net 和 Q_Net 中选择
        if self.Duel:
            self.q_net = Duel_Q_Net(opt).to(self.dvc)
        else:
            self.q_net = Q_Net(opt).to(self.dvc)
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=opt.lr)
        self.q_target = copy.deepcopy(self.q_net)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_target.parameters():
            p.requires_grad = False
        self.target_freq = opt.target_freq

    def select_action(self, state, evaluate):
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.dvc)
            # NOTE # NoisyNet时，不需要e-greedy
            if self.Noisy:
                return self.q_net(state).argmax().item()
            else:
                # NOTE: 在测试时也可以选择最低概率的小探索, 用来测试策略是否稳健
                p = 0.01 if evaluate else self.exp_noise
                if np.random.rand() < p:
                    return np.random.randint(0, self.action_dim)
                else:
                    return self.q_net(state).argmax().item()

    def train(self, replay_buffer):
        self.train_counter += 1
        s, a, r, s_next, dw = replay_buffer.sample(self.batch_size)

        """Compute the target Q value"""
        with torch.no_grad():
            if self.Double:
                argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)  # 主网络选出最优动作
                max_q_prime = self.q_target(s_next).gather(1, argmax_a)  # 目标网络计算动作的Q值
            else:
                max_q_prime = self.q_target(s_next).max(1)[0].unsqueeze(1)

            """Avoid impacts caused by reaching max episode steps"""
            target_Q = r + (~dw) * self.gamma * max_q_prime  # dw: die or win

        # Get current Q estimates
        current_q = self.q_net(s)
        current_q_a = current_q.gather(1, a)

        if self.huber_loss:
            # NOTE: HUBER_LOSS 可以降低对异常值的敏感度
            q_loss = F.huber_loss(current_q_a, target_Q)
        else:
            q_loss = F.mse_loss(current_q_a, target_Q)

        self.q_net_optimizer.zero_grad()
        q_loss.backward()
        # NOTE: 梯度裁剪, 当参数的模大于 20 时, 对整个梯度进行缩放, 而不是直接截断
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 20)
        self.q_net_optimizer.step()

        # hard target update
        if self.train_counter % self.target_freq == 0:
            for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
                target_param.data.copy_(param.data)
        for p in self.q_target.parameters():
            p.requires_grad = False

    def save(self, ExperimentName, index):
        torch.save(self.q_net.state_dict(), f"./2.2_Noisy-Duel-DDQN-Atari/model/{ExperimentName}_{index}k.pth")

    def load(self, ExperimentName, index):
        model_path = f"./2.2_Noisy-Duel-DDQN-Atari/model/{ExperimentName}_{index}k.pth"
        if not os.path.exists(model_path):
            print(f"No model {model_path} found, please train first.")
            return
        self.q_net.load_state_dict(torch.load(model_path, map_location=self.dvc))
        self.q_target.load_state_dict(torch.load(model_path, map_location=self.dvc))


class ReplayBuffer_torch:
    def __init__(self, device, max_size=int(1e5)):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # 状态变为 (4, 84, 84), 这个地方都没有设置到device上
        self.state = torch.zeros((max_size, 4, 84, 84), dtype=torch.uint8)
        self.action = torch.zeros((max_size, 1), dtype=torch.int64)
        self.reward = torch.zeros((max_size, 1))
        self.next_state = torch.zeros((max_size, 4, 84, 84), dtype=torch.uint8)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool)

    def add(self, state, action, reward, next_state, dw):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dw[self.ptr] = dw  # 0,0,0，...，1

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # ind = np.random.choice((self.size-1), batch_size, replace=False)  # Time consuming, but no duplication
        # 但是这里没有用 pytorch 进行抽样, 速度还不是最快
        ind = np.random.randint(0, (self.size - 1), batch_size)  # Time effcient, might duplicates
        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.next_state[ind].to(self.device),
            self.dw[ind].to(self.device),
        )
