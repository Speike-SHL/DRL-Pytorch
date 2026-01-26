from torch.distributions import Categorical
from utils import Actor, Critic
import numpy as np
import torch
import copy
import math


class PPO_discrete:
    def __init__(self, **kwargs):
        # Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)

        """构建 Actor 和 Critic 网络"""
        self.actor = Actor(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
        self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)

        """初始化学习率调度器"""
        self.current_lr = self.max_lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.current_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.current_lr * self.critic_lr_coef)

        """Build Trajectory holder"""
        # NOTE: PPO 中的Holder不能过长、使用后丢弃、且需要连续的轨迹。
        # https://www.wolai.com/ocv6KWvM4rhJ27Z4GH1YLa#x25BUuauQ7xJ2hMrnrb7Do
        self.s_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.a_hoder = np.zeros((self.T_horizon, 1), dtype=np.int64)
        self.r_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.s_next_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.logprob_a_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.done_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        self.dw_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)

    def update_lr(self, current_steps, max_steps):
        """根据训练进度更新学习率"""
        progress = min(current_steps / max_steps, 1.0)

        # 线性衰减：从 max_lr 衰减到 min_lr
        self.current_lr = self.max_lr - (self.max_lr - self.min_lr) * progress
        # 更新优化器的学习率
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.current_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = self.current_lr * self.critic_lr_coef

        return self.current_lr

    def select_action(self, s, deterministic):
        """根据状态选择动作
        Actor网络输出每个动作的概率
        1. 如果是确定性策略, 则选择概率最高的动作
        2. 如果是随机策略, 则从 Categorical 分布中采样动作
        Categorical 分布是通过一个离散概率建立的分布
        """
        s = torch.from_numpy(s).float().to(self.dvc)
        with torch.no_grad():
            pi = self.actor.pi(s, softmax_dim=0)
            if deterministic:
                a = torch.argmax(pi).item()
                return a, None
            else:
                m = Categorical(pi)
                a = m.sample().item()
                pi_a = pi[a].item()
                return a, pi_a

    def train(self):
        self.entropy_coef *= self.entropy_coef_decay  # exploring decay
        """Prepare PyTorch data from Numpy data"""
        s = torch.from_numpy(self.s_hoder).to(self.dvc)
        a = torch.from_numpy(self.a_hoder).to(self.dvc)
        r = torch.from_numpy(self.r_hoder).to(self.dvc)
        s_next = torch.from_numpy(self.s_next_hoder).to(self.dvc)
        old_prob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
        done = torch.from_numpy(self.done_hoder).to(self.dvc)
        dw = torch.from_numpy(self.dw_hoder).to(self.dvc)

        """ Use TD+GAE+LongTrajectory to compute Advantage and TD target"""
        with torch.no_grad():
            vs = self.critic(s)  # 当前状态价值估计
            vs_ = self.critic(s_next)  # 下一状态价值估计

            """dw(dead and win) for TD_target and Adv"""
            deltas = r + self.gamma * vs_ * (~dw) - vs  # 计算单步的 TD-error = TD-target - TD-value
            deltas = deltas.cpu().flatten().numpy()

            """反向遍历并递归计算 GAE """
            adv = [0]  # 储存优势, 末尾先加一个0, 便于递归
            for dlt, done in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):  # dlt 表示 ẟt
                # At = ẟt + γ * λ * (1 - done) * At+1
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                adv.append(advantage)
            # 纠正优势的顺序并舍弃最后一个0, 因为最后一个0是用于递归的初始值
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)  # (2048,1)
            # 利用计算好的优势函数, 加上当前状态价值，得到 TD-target。实际上也是 Q 值？
            # 参考 https://www.zhihu.com/question/626325093
            td_target = adv + vs
            # 对优势函数进行归一化, 帮助 Actor 训练更加稳定
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  # sometimes helps

        """PPO update"""
        # 将长轨迹切分为短轨迹，并执行小批量 PPO 更新
        optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))
        # 记录损失值和熵值
        total_a_loss = 0
        total_c_loss = 0
        total_entropy = 0
        total_loss_clip = 0

        for _ in range(self.K_epochs):
            # 随机打乱轨迹顺序, 帮助模型学习到更通用的策略
            perm = np.arange(s.shape[0])  # permutation 排列的缩写, 生成 0 - 2048 的索引
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.dvc)  # LongTensor 确保转为 tensor 后不是变为float类型
            s, a, td_target, adv, old_prob_a = (  # 加 clone 确保不会污染计算图
                s[perm].clone(),
                a[perm].clone(),
                td_target[perm].clone(),
                adv[perm].clone(),
                old_prob_a[perm].clone(),
            )

            """mini-batch PPO update"""
            for i in range(optim_iter_num):
                # 获取该批次数据的索引
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))

                """actor update"""
                # 当前策略下每个动作的概率, 维度为(batch_size, action_dim), 因此 softmax_dim 为1
                prob = self.actor.pi(s[index], softmax_dim=1)
                # 生成 batch_size 个分布, 计算每个分布的熵(batch_size), 并求和, 最终维度为(1)
                entropy = Categorical(prob).entropy().sum(0, keepdim=True)
                # 新策略中执行当前动作的概率 pi(a|s)
                prob_a = prob.gather(1, a[index])
                # 重要性采样比率, pi(a|s) / pi_old(a|s)
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a[index]))

                # surrogate loss
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                # 加上负号因为 PPO 中优化目标是最大化期望回报,即在状态s下执行动作a,得到的优势比平均动作好多少,
                # 而pytorch 中是求最小化
                loss_clip = -torch.min(surr1, surr2)
                # 当策略趋于确定, 熵会减小, 乘上负号表示loss变大, 帮助模型保持探索
                loss_entropy = -self.entropy_coef * entropy
                a_loss = loss_clip + loss_entropy  # (batch_size,1)

                # 清空梯度、反向传播、裁剪梯度, 更新参数
                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                """critic update"""
                # 最小化状态价值预测的均方根误差。为什么和 Q 值求 loss？
                # 因为 PPO 在采样动作时采到的是概率最大的动作, 即大概率在状态 s 情况下 agent 会执行动作 a, 获得当前奖励 r
                # 这样 Q 值就比网络预测的 V 值更接近当前的实际状态价值
                # 参考 https://www.zhihu.com/question/626325093
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                # 遍历网络每一层, 如果是权重参数, 则利用该层的所有权重计算L2正则化项,
                # 用于防止过拟合
                for name, param in self.critic.named_parameters():
                    if "weight" in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()  # 这里没mean, 因为计算c_loss时已经mean过了
                self.critic_optimizer.step()

                # 累加损失值和熵值
                total_a_loss += a_loss.mean().item()
                total_c_loss += c_loss.item()
                total_entropy += entropy.item()
                total_loss_clip += loss_clip.mean().item()

        # 计算平均值
        avg_a_loss = total_a_loss / (self.K_epochs * optim_iter_num)
        avg_c_loss = total_c_loss / (self.K_epochs * optim_iter_num)
        avg_entropy = total_entropy / (self.K_epochs * optim_iter_num)
        avg_loss_clip = total_loss_clip / (self.K_epochs * optim_iter_num)

        return avg_a_loss, avg_c_loss, avg_entropy, avg_loss_clip

    def put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
        self.s_hoder[idx] = s
        self.a_hoder[idx] = a
        self.r_hoder[idx] = r
        self.s_next_hoder[idx] = s_next
        self.logprob_a_hoder[idx] = logprob_a
        self.done_hoder[idx] = done
        self.dw_hoder[idx] = dw

    def save(self, episode):
        torch.save(
            self.critic.state_dict(), "./3.1 PPO-Discrete/model/ppo_critic_{}_{}.pth".format(self.EnvIdex, episode)
        )
        torch.save(
            self.actor.state_dict(), "./3.1 PPO-Discrete/model/ppo_actor_{}_{}.pth".format(self.EnvIdex, episode)
        )

    def load(self, episode):
        self.critic.load_state_dict(
            torch.load(
                "./3.1 PPO-Discrete/model/ppo_critic_{}_{}.pth".format(self.EnvIdex, episode), map_location=self.dvc
            )
        )
        self.actor.load_state_dict(
            torch.load(
                "./3.1 PPO-Discrete/model/ppo_actor_{}_{}.pth".format(self.EnvIdex, episode), map_location=self.dvc
            )
        )
