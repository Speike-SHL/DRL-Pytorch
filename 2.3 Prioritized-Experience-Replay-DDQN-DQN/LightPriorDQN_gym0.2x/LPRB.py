import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LightPriorReplayBuffer:
    """
    缓冲区按时间顺序存, 无需显式保存 s_next，更加节省内存，尤其对图像状态友好。

    迭代时，按如下方式添加新 transition：
        a = model.select(s)
        s_next, r, dw, tr, info = env.step(a)
        buffer.add(s, a, r, dw, tr)
        # dw: 表示 's_next' 是否为终止状态
        # tr: 表示该回合是否被截断。

        s_next 会在下一次 add 时作为新的 s 被存入 buffer。

    采样时，
    需避免 ind = [ptr - 1] 和 ind = [size - 1]，因为不是这一episode的数据。
    以保证 state[ind] 与 state[ind+1] 的连续性。
    然后，
    s = self.state[ind]
    s_next = self.state[ind+1]

    dr代表回合终止，tr代表回合截断。
    重要的是，由于我们没有显式保存 's_next'，当 dw 或 tr 为 True 时，s[ind] 与 s[ind+1] 并不来自同一回合。
    当遇到 dw=True 时，
    self.state[ind+1] 并不是 self.state[ind] 真正的下一状态，而是一个新重置后的状态。
    这没关系，因为 Q_target[s[ind],a[ind]] = r[ind] + gamma*(1-dw[ind])* max_Q(s[ind+1],·)，
    当 dw=true 时，我们根本不会用到 s[ind+1]。
    然而，当遇到 tr=True 时，
    self.state[ind+1] 并不是 self.state[ind] 真正的下一状态，而是一个新重置后的状态，
    因此必须在损失函数里通过 (1-tr) 丢弃该 transition。

    于是，在训练时，
    Q_target = r + self.gamma * (1-dw) * max_q_next
    current_Q = self.q_net(s).gather(1,a)
    q_loss = torch.square((1-tr) * (current_Q - Q_target)).mean()

    """

    def __init__(self, opt):
        self.device = device

        self.ptr = 0  # 下一元素应该在的位置
        self.size = 0  # 当前 buffer 的大小

        self.state = torch.zeros((opt.buffer_size, opt.state_dim), device=device)  # 如果是图像，可以用unit8节省空间
        self.action = torch.zeros((opt.buffer_size, 1), dtype=torch.int64, device=device)
        self.reward = torch.zeros((opt.buffer_size, 1), device=device)
        self.dw = torch.zeros((opt.buffer_size, 1), dtype=torch.bool, device=device)  # only 0/1 回合终止
        self.tr = torch.zeros((opt.buffer_size, 1), dtype=torch.bool, device=device)  # only 0/1 回合截断
        # 和书上不一样, 这里 alpha 控制了优先级的尖锐程度, alpha = 0 时无优先级采样, alpha = 1 时完全按优先级采样, 通常取 0~1 之间的值
        self.priorities = torch.zeros(opt.buffer_size, dtype=torch.float32, device=device)  # (|TD-error|+0.01)^alpha
        self.buffer_size = opt.buffer_size

        self.alpha = opt.alpha
        self.beta = opt.beta_init
        self.replacement = opt.replacement  # 是否有放回采样

    def add(self, state, action, reward, dw, tr, priority):
        self.state[self.ptr] = torch.from_numpy(state).to(device)
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.dw[self.ptr] = dw
        self.tr[self.ptr] = tr
        self.priorities[self.ptr] = priority

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        # 因为state[ptr-1]和state[ptr]，state[size-1]和state[size]不来自同一个episode
        # 所以从[0, size-1)中sample; 这里必须clone
        # 当缓冲区没满时, size为当前存储的元素个数, size=ptr, 此时size-1是最后一个状态, 它的下一个状态是不存在的, 所以不能采样到size-1
        Prob_torch_gpu = self.priorities[0 : self.size - 1].clone()
        if self.ptr < self.size:
            # 当ptr<size时, 说明缓冲区已满, ptr的指向和ptr-1不是一个episode, 因此不能取到ptr-1
            # 但是ptr-1之前的和ptr之后的都是可以取的
            Prob_torch_gpu[self.ptr - 1] = 0  # 并且不能包含ptr-1
        ind = torch.multinomial(
            Prob_torch_gpu, num_samples=batch_size, replacement=self.replacement
        )  # replacement=True数据可能重复，但是快很多; (batchsize,)
        # 注意，这里的ind对于self.priorities和Prob_torch_gpu是通用的，并没有错位

        # beta 用于修正优先级采样带来的偏差, 初期比较小, 表示不做修正, 逐渐增大到1来保证收敛性
        IS_weight = (self.size * Prob_torch_gpu[ind]) ** (-self.beta)
        # 归一化, 如果某些样本采样概率非常小, 那么IS_weight会非常大, 导致loss爆炸
        Normed_IS_weight = (IS_weight / IS_weight.max()).unsqueeze(-1)  # (batchsize,1)

        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.state[ind + 1],
            self.dw[ind],
            self.tr[ind],
            ind,
            Normed_IS_weight,
        )
