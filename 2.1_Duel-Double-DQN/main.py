from utils import evaluate_policy, str2bool
from datetime import datetime
from DQN import DQN_agent
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import shutil
import argparse
import torch


"""Hyperparameter Setting"""
parser = argparse.ArgumentParser()
parser.add_argument("--dvc", type=str, default="cuda", help="running device: cuda or cpu")
parser.add_argument("--EnvIdex", type=int, default=1, help="CartPole-v1, LunarLander-v3")
parser.add_argument("--write", type=str2bool, default=True, help="Use SummaryWriter to record the training")
parser.add_argument(
    "--render", type=str2bool, default=False, help="If True, no training, just evaluate. Thus you should loadmodel."
)
parser.add_argument("--Loadmodel", type=str2bool, default=False, help="Load pretrained model or Not")
parser.add_argument("--ModelIdex", type=int, default=100, help="how many steps model should we load, in thousands")
parser.add_argument("--record_video", type=str2bool, default=True, help="Record video every eval_interval.")

parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--Max_train_steps", type=int, default=int(1e6), help="Max training steps")
parser.add_argument("--save_interval", type=int, default=int(5e4), help="Model saving interval, in steps.")
parser.add_argument("--eval_interval", type=int, default=int(2e3), help="Model evaluating interval, in steps.")
parser.add_argument("--random_steps", type=int, default=int(3e3), help="before random_steps, only exploration")
parser.add_argument(
    "--update_every",
    type=int,
    default=50,
    help="every update_every steps, update update_every times. rather than 1 training per step.",
)

parser.add_argument("--gamma", type=float, default=0.99, help="Discounted Factor")
parser.add_argument("--net_width", type=int, default=200, help="Hidden net width")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=256, help="lenth of sliced trajectory")
parser.add_argument("--exp_noise", type=float, default=0.2, help="explore noise in epsilon-greedy")
parser.add_argument("--noise_decay", type=float, default=0.99, help="decay rate of explore noise")
parser.add_argument("--Double", type=str2bool, default=True, help="Whether to use Double Q-learning")
parser.add_argument("--Duel", type=str2bool, default=True, help="Whether to use Duel networks")
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc if torch.cuda.is_available() else "cpu")  # from str to torch.device
print(opt)


def main():
    EnvName = ["CartPole-v1", "LunarLander-v3"]
    BriefEnvName = ["CPV1", "LLdV3"]
    env = gym.make(EnvName[opt.EnvIdex], render_mode="human" if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex], render_mode="rgb_array" if opt.record_video else None)
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps

    # Algorithm Setting
    # TODO: 改为叠加的，不用DDQN
    if opt.Duel:
        algo_name = "Duel"
    else:
        algo_name = ""
    if opt.Double:
        algo_name += "DDQN"
    else:
        algo_name += "DQN"

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True  # 强制使用固定算法
    torch.backends.cudnn.benchmark = False  # 关闭动态性能优化
    print("Random Seed: {}".format(opt.seed))

    print(
        "Algorithm:",
        algo_name,
        "  Env:",
        BriefEnvName[opt.EnvIdex],
        "  state_dim:",
        opt.state_dim,
        "  action_dim:",
        opt.action_dim,
        "  Random Seed:",
        opt.seed,
        "  max_e_steps:",
        opt.max_e_steps,
        "\n",
    )

    writepath = None
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter

        timenow = str(datetime.now())[0:-10]
        timenow = " " + timenow[0:13] + "_" + timenow[-2::]
        writepath = (
            "./2.1_Duel-Double-DQN/runs/{}-{}_S{}_".format(algo_name, BriefEnvName[opt.EnvIdex], opt.seed) + timenow
        )
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    if opt.record_video:
        eval_env = RecordVideo(eval_env, video_folder=writepath, episode_trigger=lambda x: x % 3 == 0)

    # Build model and replay buffer
    if not os.path.exists("./2.1_Duel-Double-DQN/model"):
        os.mkdir("./2.1_Duel-Double-DQN/model")
    agent = DQN_agent(**vars(opt))  # 传入opt的所有参数
    if opt.Loadmodel:
        if not agent.load(algo_name, BriefEnvName[opt.EnvIdex], opt.ModelIdex):
            return

    if opt.render:
        while True:
            score = evaluate_policy(env, agent, 1)
            print("EnvName:", BriefEnvName[opt.EnvIdex], "seed:", opt.seed, "score:", score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            # NOTE: 前面固定其他部分的随机种子是为了固定算法随机性, 但这里需要不同的种子来训练Agent, 避免过拟合
            # NOTE: 这是和其他实现不同的地方
            s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1  # 每个episode的seed不同，避免overfit
            done = False

            """Interact & train, 每回合"""
            while not done:
                # e-greedy exploration
                if total_steps < opt.random_steps:  # random_steps 前纯探索
                    a = env.action_space.sample()
                else:  # 后面再用 Ɛ-greedy 边探索边利用
                    a = agent.select_action(s, deterministic=False)
                s_next, r, dw, tr, info = env.step(a)  # dw: dead&win; tr: truncated
                done = dw or tr

                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next

                """Update"""
                # NOTE：每 50 步进行 50 次训练比每 1 步训练 1 次更好！
                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        agent.train()

                """Noise decay & Record & Log"""
                if total_steps % 1000 == 0:
                    agent.exp_noise *= opt.noise_decay
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, turns=3)
                    if opt.write:
                        writer.add_scalar("ep_r", score, global_step=total_steps)
                        writer.add_scalar("noise", agent.exp_noise, global_step=total_steps)
                    print(
                        "EnvName:",
                        BriefEnvName[opt.EnvIdex],
                        "seed:",
                        opt.seed,
                        "steps: {}k".format(int(total_steps / 1000)),
                        "score:",
                        int(score),
                        "record_video:",
                        f"rl-video-episode-{int(total_steps / 2000)*3}.mp4",
                    )
                total_steps += 1

                """save model"""
                if total_steps % opt.save_interval == 0:
                    agent.save(algo_name, BriefEnvName[opt.EnvIdex], int(total_steps / 1000))
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
