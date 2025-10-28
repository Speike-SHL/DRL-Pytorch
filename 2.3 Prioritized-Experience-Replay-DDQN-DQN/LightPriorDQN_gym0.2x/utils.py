import argparse


def evaluate_policy(env, model, turns=3):
    scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)  # dw: terminated; tr: truncated
            done = dw + tr
            scores += r
            s = s_next
    return int(scores / turns)


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, initial_p, final_p):
        """在 schedule_timesteps 步数内，从 initial_p 到 final_p 进行线性插值。
        超过该步数后，始终返回 final_p。
        参数
        ----------
        schedule_timesteps: int
            线性退火 initial_p 到 final_p 的步数
        initial_p: float
            初始输出值
        final_p: float
            最终输出值
        """
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


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
        raise argparse.ArgumentTypeError("Boolean value expected.")
