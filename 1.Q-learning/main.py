from Q_learning import QLearningAgent, evaluate_policy
from torch.utils.tensorboard import SummaryWriter  # used to plot training curve
from gymnasium.wrappers import TimeLimit
from datetime import datetime
import gymnasium as gym
import numpy as np
import os
import shutil


def main():
    write = True  # whether use SummaryWriter to record training curve
    Loadmodel = False  # Load model or not
    Max_train_steps = 20000
    seed = 0
    np.random.seed(seed)
    print(f"Random Seed: {seed}")

    """ ↓↓↓ Build Env ↓↓↓ """
    EnvName = "CliffWalking-v1"
    env = gym.make(EnvName)
    env = TimeLimit(env, max_episode_steps=500)
    eval_env = gym.make(EnvName)
    eval_env = TimeLimit(eval_env, max_episode_steps=100)

    """ ↓↓↓ Use tensorboard to record training curves ↓↓↓ """
    if write:
        # Use SummaryWriter to record the trainig
        timenow = str(datetime.now())[0:-7]  # 去掉最后的毫秒
        timenow = " " + timenow[0:13] + "_" + timenow[14:16] + "_" + timenow[-2::]  # 整理格式
        writepath = "./1.Q-learning/runs/{}".format(EnvName) + timenow
        # 如果路径已存在，执行递归删除
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    """ ↓↓↓ Build Q-learning Agent ↓↓↓ """
    if not os.path.exists("./1.Q-learning/model"):
        os.mkdir("./1.Q-learning/model")
    # observation_space.n 是离散空间的状态数，而 observation_space.shape[0] 是连续空间的状态维度
    agent = QLearningAgent(s_dim=env.observation_space.n, a_dim=env.action_space.n, lr=0.2, gamma=0.9, exp_noise=0.1)
    if Loadmodel:
        agent.restore()

    """ ↓↓↓ Iterate and Train ↓↓↓ """
    total_steps = 0
    # 限制的不是回合数，是总步数？
    while total_steps < Max_train_steps:
        s, info = env.reset(seed=seed)
        seed += 1  # 随机种子在变
        done, steps = False, 0

        # 每个回合的训练
        while not done:
            steps += 1
            # 动作选择是使用确定性的每次最大Q值(用于测试), 还是使用 Ɛ - 贪婪策略
            a = agent.select_action(s, deterministic=False)
            s_next, r, dw, tr, info = env.step(a)
            agent.train(s, a, r, s_next, dw)

            done = dw or tr
            s = s_next

            total_steps += 1
            """record & log"""
            if total_steps % 100 == 0:
                ep_r = evaluate_policy(eval_env, agent)  # episode reward, 每 100 步评估一次
                if write:
                    # 数据标签, 标量值(y), 当前步数(x)
                    writer.add_scalar("ep_r", ep_r, global_step=total_steps)
                print(f"EnvName:{EnvName}, Seed:{seed}, Steps:{total_steps}, Episode reward:{ep_r}")

            """save model"""
            if total_steps % Max_train_steps == 0:
                agent.save()

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
