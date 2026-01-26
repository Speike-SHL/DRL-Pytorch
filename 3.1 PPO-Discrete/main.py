from utils import evaluate_policy, str2bool
from datetime import datetime
from PPO import PPO_discrete
import gymnasium as gym
import os
import argparse
import torch
import wandb

# os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
os.environ["WANDB_DIR"] = os.path.join(os.path.dirname(__file__), "runs")
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_INIT_TIMEOUT"] = "120"
os.environ["WANDB_API_KEY"] = "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"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

"""Hyperparameter Setting"""
parser = argparse.ArgumentParser()
parser.add_argument("--dvc", type=str, default="cuda", help="running device: cuda or cpu")
parser.add_argument("--EnvIdex", type=int, default=1, help="CP-v1, LLd-v3")
parser.add_argument("--mode", type=str, default="train", help="run mode: train, eval, sweep")
parser.add_argument("--write", type=str2bool, default=False, help="Use wandb to record the training")
parser.add_argument(
    "--render", type=str2bool, default=False, help="If True, no training, just evaluate. Thus you should loadmodel."
)
parser.add_argument("--Loadmodel", type=str2bool, default=False, help="Load pretrained model or Not")
parser.add_argument("--ModelIdex", type=int, default=800000, help="which model to load")

parser.add_argument("--seed", type=int, default=5, help="random seed")
parser.add_argument(
    "--T_horizon", type=int, default=2048, help='智能体在单次训练前收集的轨迹的最大时间步长度 ，即"轨迹的时间视野"'
)
parser.add_argument("--Max_train_steps", type=int, default=8e5, help="Max training steps")
parser.add_argument("--save_interval", type=int, default=4e5, help="Model saving interval, in steps.")
parser.add_argument("--eval_interval", type=int, default=1e4, help="Model evaluating interval, in steps.")

parser.add_argument("--gamma", type=float, default=0.99, help="Discounted Factor")
parser.add_argument("--lambd", type=float, default=0.95, help="GAE Factor")
parser.add_argument("--clip_rate", type=float, default=0.2, help="PPO Clip rate")
parser.add_argument("--K_epochs", type=int, default=15, help="PPO update times, 用一批数据进行多次训练")
parser.add_argument("--net_width", type=int, default=128, help="Hidden net width")
parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate (initial)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (final)")
parser.add_argument(
    "--critic_lr_coef",
    type=float,
    default=10.0,
    help="Critic learning rate coefficient (critic_lr = actor_lr * critic_lr_coef)",
)
parser.add_argument("--l2_reg", type=float, default=0.001, help="L2 regulization coefficient for Critic")
parser.add_argument("--batch_size", type=int, default=64, help="lenth of sliced trajectory")
parser.add_argument("--entropy_coef", type=float, default=0.001, help="Entropy coefficient of Actor")
parser.add_argument("--entropy_coef_decay", type=float, default=0.999, help="Decay rate of entropy_coef")
parser.add_argument("--adv_normalization", type=str2bool, default=False, help="Advantage normalization")
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)  # from str to torch.device
print(opt)


def main():
    # Build Training Env and Evaluation Env
    EnvName = ["CartPole-v1", "LunarLander-v3"]
    BriefEnvName = ["CP-v1", "LLd-v3"]

    # Use wandb to record training curves
    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = " " + timenow[0:13] + "_" + timenow[-2::]
        if opt.mode == "sweep":
            run = wandb.init(
                entity="shao_haoluo-hust",
                project="3.1 PPO-Discrete",
                name=timenow.strip(),
                group=BriefEnvName[opt.EnvIdex],
                tags=[],
                notes="",
                dir="./3.1 PPO-Discrete/runs/",
                job_type="training",
                reinit="finish_previous",
            )
            # 从 run.confg 中更新 opt 的参数
            opt.__dict__.update(run.config)
        else:
            run = wandb.init(
                entity="shao_haoluo-hust",
                project="3.1 PPO-Discrete",
                name=timenow.strip(),
                group=BriefEnvName[opt.EnvIdex],
                tags=[],
                notes="",
                dir="./3.1 PPO-Discrete/runs/",
                job_type="training",
                reinit="finish_previous",
                config=opt,
            )
        print(opt)

    # 创建训练环境和评估环境
    env = gym.make(EnvName[opt.EnvIdex], render_mode="human" if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])  # 评估环境, 可以把评估环境设置为渲染模式
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True  # 强制使用固定算法
    torch.backends.cudnn.benchmark = False  # 关闭动态性能优化
    print("Random Seed: {}".format(opt.seed))

    print(
        "Env:",
        BriefEnvName[opt.EnvIdex],
        "  state_dim:",
        opt.state_dim,
        "  action_dim:",
        opt.action_dim,
        "   Random Seed:",
        opt.seed,
        "  max_e_steps:",
        opt.max_e_steps,
    )
    print("\n")

    if not os.path.exists("./3.1 PPO-Discrete/model"):
        os.mkdir("./3.1 PPO-Discrete/model")
    # 创建 PPO 智能体并传入参数
    agent = PPO_discrete(**vars(opt))
    # 加载预训练模型
    if opt.Loadmodel:
        agent.load(opt.ModelIdex)

    if opt.render:
        ep_r = evaluate_policy(env, agent, turns=10)
        print(f"Env:{EnvName[opt.EnvIdex]}, Episode Reward:{ep_r}")
    else:
        traj_lenth, total_steps = 0, 0
        if opt.write:
            wandb.define_metric("*", step_metric="total_steps")
        while total_steps < opt.Max_train_steps:
            # NOTE: 前面固定其他部分的随机种子是为了固定算法随机性, 但这里需要不同的种子来训练Agent, 避免过拟合
            # NOTE: 这是和其他实现不同的地方
            s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1  # 每个episode的seed不同，避免overfit
            done = False

            """Interact & trian, 每回合"""
            episode_reward = 0
            episode_length = 0
            while not done:
                """Interact with Env"""
                # NOTE: 为什么不用 episilon-greedy？
                # https://www.wolai.com/ocv6KWvM4rhJ27Z4GH1YLa#jnJRetJL1XLU4iw4J2JEtz
                a, logprob_a = agent.select_action(s, deterministic=False)  # use stochastic when training
                s_next, r, dw, tr, info = env.step(a)  # dw: dead&win; tr: truncated
                # TODO: 尝试不截断的效果
                # NOTE: 截断LunarLander环境中坠毁时很大的负奖励。避免Agent不敢探索, 同时使训练过程更加平滑
                if r <= -100:
                    r = -30  # good for LunarLander
                done = dw or tr  # done 是指所有终止情况

                episode_reward += r
                episode_length += 1

                """Store the current transition"""
                agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx=traj_lenth)
                s = s_next

                traj_lenth += 1
                total_steps += 1

                """每 T_horizon 轨迹长度训练一次"""
                if traj_lenth % opt.T_horizon == 0:
                    current_lr = agent.update_lr(total_steps, opt.Max_train_steps)
                    a_loss, c_loss, entropy, loss_clip = agent.train()
                    traj_lenth = 0
                    # 记录损失值和熵值
                    if opt.write:
                        wandb.log(
                            {
                                "a_loss": a_loss,
                                "c_loss": c_loss,
                                "entropy": entropy,
                                "loss_clip": loss_clip,
                                "learning_rate": current_lr,
                                "total_steps": total_steps,
                            }
                        )

                """Record & log"""
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(
                        eval_env, agent, turns=2
                    )  # evaluate the policy for 2 times, and get averaged result
                    if opt.write:
                        wandb.log({"ep_r": score, "total_steps": total_steps})
                    print(
                        "EnvName:",
                        EnvName[opt.EnvIdex],
                        "seed:",
                        opt.seed,
                        "steps: {}k".format(int(total_steps / 1000)),
                        "score:",
                        score,
                    )

                """Save model"""
                if total_steps % opt.save_interval == 0:
                    agent.save(total_steps)

            # Episode 结束，记录 episode 指标
            if opt.write:
                wandb.log(
                    {"episode_reward": episode_reward, "episode_length": episode_length, "total_steps": total_steps}
                )

        env.close()
        eval_env.close()
        wandb.finish()


def run_sweep():
    """Run wandb sweep for hyperparameter tuning"""
    sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "ep_r"},
        "parameters": {
            "T_horizon": {"values": [1024, 2048, 3072, 4096]},
            "gamma": {"values": [0.99, 0.95, 0.90]},
            "lambd": {"values": [1, 0.95, 0.60, 0.20, 0.05]},
            "clip_rate": {"values": [0.1, 0.2, 0.5, 1]},
            "K_epochs": {"values": [10, 20, 30]},
            "net_width": {"values": [64, 128, 256, 512]},
            "l2_reg": {"values": [0.1, 0.001, 0.00001]},
            "batch_size": {"values": [64, 128, 256]},
            "entropy_coef": {"values": [0.1, 0.01, 0.001]},
            "entropy_coef_decay": {"values": [1, 0.999, 0.90, 0.80, 0.50]},
            "adv_normalization": {"values": [True, False]},
            "critic_lr_coef": {"values": [0.1, 1, 10, 100]},
        },
    }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="3.1 PPO-Discrete")
    sweep_id = "shao_haoluo-hust/3.1 PPO-Discrete/yanhnoxs"
    wandb.agent(sweep_id, function=main, count=30)


if __name__ == "__main__":
    if opt.mode == "sweep":
        run_sweep()
    else:
        main()
