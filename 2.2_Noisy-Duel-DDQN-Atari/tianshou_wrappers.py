# Borrow a lot from tianshou:
# https://github.com/thu-ml/tianshou/blob/master/examples/atari/atari_wrapper.py
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import cv2
import ale_py

"""
类似于 gymnasium 中的包装器, 用于逐层包装 Atari 环境, 实现更多的功能
"""


def make_env_tianshou(
    env_name, noop_reset=True, episode_life=True, clip_rewards=True, frame_stack=4, warp_frame=True, render_mode=None
):
    """按照 DeepMind 风格配置 Atari 环境。
    同时支持 Gymnasium (s,r,term,trunc,info) 与 Gym (s,r,done,info) 两种 API。
    观测形状为 (4, 84, 84)，数据类型为 torch.uint8，返回类型为 <class 'torch.Tensor'>。
    # 此处不对观测做 (0,1) 浮点归一化，而是使用 uint8 以节省内存。
    """
    assert "NoFrameskip" in env_name
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode=render_mode)
    if noop_reset:
        env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if warp_frame:
        env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, frame_stack)
    return env


def _parse_reset_result(reset_result):
    contains_info = isinstance(reset_result, tuple) and len(reset_result) == 2 and isinstance(reset_result[1], dict)
    if contains_info:
        return reset_result[0], reset_result[1], contains_info
    return reset_result, {}, contains_info


class NoopResetEnv(gym.Wrapper):
    """在 reset 时通过执行随机数量的 no-op 来采样初始状态。用于增加初始状态的多样性
    假设 no-op 动作为 0。
    :param gym.Env env: 要包装的环境。
    :param int noop_max: 运行的最大 no-op 数量。
    """

    def __init__(self, env, noop_max=30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        _, info, return_info = _parse_reset_result(self.env.reset(**kwargs))
        if hasattr(self.unwrapped.np_random, "integers"):
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            step_result = self.env.step(self.noop_action)
            if len(step_result) == 4:
                obs, rew, done, info = step_result
            else:
                obs, rew, term, trunc, info = step_result
                done = term or trunc
            if done:
                obs, info, _ = _parse_reset_result(self.env.reset())
        if return_info:
            return obs, info
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """仅返回每 `skip` 帧中的最新原始观测（用于在时序上做最大池化）。
    可以减少计算量，提高训练效率
    :param gym.Env env: 要包装的环境。
    :param int skip: 跳帧的数量。
    """

    def __init__(self, env, skip=4) -> None:
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Step the environment with the given action.

        Repeat action, sum reward, and max over last observations.
        """
        obs_list, total_reward = [], 0.0
        new_step_api = False
        for _ in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, term, trunc, info = step_result
                done = term or trunc
                new_step_api = True
            obs_list.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(obs_list[-2:], axis=0)
        if new_step_api:
            return max_frame, total_reward, term, trunc, info

        return max_frame, total_reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    """将“失去生命”视为回合结束，但只在真正的游戏结束时(生命耗尽时)才重置环境。
    这有助于价值函数的估计。
    :param gym.Env env: 要包装的环境。
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        self._return_info = False

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            new_step_api = False
        else:
            obs, reward, term, trunc, info = step_result
            done = term or trunc
            new_step_api = True

        self.was_real_done = done
        # check current lives, make loss of life terminal, then update lives to
        # handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few
            # frames, so its important to keep lives > 0, so that we only reset
            # once the environment is actually done.
            done = True
            term = True
        self.lives = lives
        if new_step_api:
            return obs, reward, term, trunc, info
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Calls the Gym environment reset, only when lives are exhausted.

        This way all states are still reachable even though lives are episodic, and
        the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info, self._return_info = _parse_reset_result(self.env.reset(**kwargs))
        else:
            # no-op step to advance from terminal/lost life state
            step_result = self.env.step(0)
            obs, info = step_result[0], step_result[-1]
        self.lives = self.env.unwrapped.ale.lives()
        if self._return_info:
            return obs, info
        return obs


class FireResetEnv(gym.Wrapper):
    """在 reset 时主动执行“开火”动作，用于那些需要开火才能开始游戏的环境。
    相关讨论：https://github.com/openai/baselines/issues/240。
    :param gym.Env env: 要包装的环境。
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        _, _, return_info = _parse_reset_result(self.env.reset(**kwargs))
        obs = self.env.step(1)[0]
        return (obs, {}) if return_info else obs


class WarpFrame(gym.ObservationWrapper):
    """将帧缩放到 84×84，正如 Nature 论文及后续工作中所采用的方式。
    大幅减少输入维度、提高训练效率
    :param gym.Env env: 要包装的环境。
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.size = 84
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(self.size, self.size),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame):
        """Returns the current observation from a frame."""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ClipRewardEnv(gym.RewardWrapper):
    """将奖励通过其符号裁剪为 {+1, 0, -1}。
    用于稳定训练过程
    :param gym.Env env: 要包装的环境。
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.reward_range = (-1, 1)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign. Note: np.sign(0) == 0."""
        return np.sign(reward)


class FrameStack(gym.Wrapper):
    """堆叠最近 n_frames 帧。
    目的：让智能体能够感知物体的运动信息
    :param gym.Env env: 要包装的环境。
    :param int n_frames: 要堆叠的帧数。
    """

    def __init__(self, env, n_frames) -> None:
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shape = (n_frames, *env.observation_space.shape)
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info, return_info = _parse_reset_result(self.env.reset(**kwargs))
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return (self._get_ob(), info) if return_info else self._get_ob()

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            new_step_api = False
        else:
            obs, reward, term, trunc, info = step_result
            new_step_api = True
        self.frames.append(obs)
        if new_step_api:
            return self._get_ob(), reward, term, trunc, info
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        """Note that here is different from original Tianshou Wrapper"""
        # the original wrapper use `LazyFrames` but since we use np buffer, it has no effect
        return torch.tensor(np.stack(self.frames, axis=0), dtype=torch.uint8)  # return torch.tensor instead of numpy
