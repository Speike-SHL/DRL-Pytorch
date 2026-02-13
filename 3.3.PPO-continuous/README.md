# PPO-连续动作空间
这是一个简洁的PyTorch实现的PPO算法，适用于连续动作空间，包含10个优化技巧。<br />

## 10个优化技巧
技巧1——优势归一化(Advantage Normalization)<br />
技巧2——状态归一化(State Normalization)<br />
技巧3和技巧4——奖励归一化与奖励缩放(Reward Normalization & Reward Scaling)<br />
技巧5——策略熵(Policy Entropy)<br />
技巧6——学习率衰减(Learning Rate Decay)<br />
技巧7——梯度裁剪(Gradient Clip)<br />
技巧8——正交初始化(Orthogonal Initialization)<br />
技巧9——Adam优化器Epsilon参数(Adam Optimizer Epsilon Parameter)<br />
技巧10——Tanh激活函数(Tanh Activation Function)<br />

## 如何使用代码？
您可以直接在IDE中运行'PPO_continuous_main.py'文件。<br />

## 训练环境
您可以在代码中设置'env_index'来切换训练环境。我们在4个环境中训练了代码。<br />
env_index=0 代表 'BipedalWalker-v3'（双足行走机器人）<br />
env_index=1 代表 'HalfCheetah-v2'（半猎豹）<br />
env_index=2 代表 'Hopper-v2'（跳跃机器人）<br />
env_index=3 代表 'Walker2d-v2'（2D行走机器人）<br />

## 训练结果
![image](https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/5.PPO-continuous/training_result.png)

## 教程
更多详细信息请参考知乎博客：https://zhuanlan.zhihu.com/p/512327050
