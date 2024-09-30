from Environment import Environment
from RL_DQN import DQN, ReplayBuffer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# GPU运算
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

# ------------------------------- #
# 全局变量
# ------------------------------- #

capacity = 500  # 经验池容量
lr = 1e-3  # 学习率
gamma = 0.98  # 折扣因子
epsilon = 0.9  # 贪心系数
epsilon_min = 0.1 #最小贪心系数
epsilon_decay = 0.98 #贪心系数折扣率
target_update = 50  # 目标网络的参数的更新频率
batch_size = 32
n_hidden = 64  # 隐含层神经元个数
min_size = 200  # 经验池超过多少后再训练
return_list = []  # 记录每个回合的回报
loss = []
loss_list = [] #用来保留每回合平均loss
Is_train = True  #是否有训练模型
episode = 1000


# 加载环境
env = Environment()
n_states = 4  # 4
n_actions = 50

# 实例化经验池
replay_buffer = ReplayBuffer(capacity)
# 实例化DQN
agent = DQN(n_states=n_states,
            n_hidden=n_hidden,
            n_actions=n_actions,
            learning_rate=lr,
            gamma=gamma,
            epsilon=epsilon,
            target_update=target_update,
            device=device,
            Is_train=Is_train
            )
for i in range(episode):
    loss = []
    return_list = []  # 记录每个回合的回报
    with tqdm(total=50) as pbar:
        # 训练模型
        for Id in range(50):
            # 每个回合开始前重置环境
            env.make(Id)
            state = env.reset()  # len=4
            # 记录每个回合的回报
            episode_return = 0
            # 记录经过的点
            action_list = [state[0]]

            done = False

            while True:
                # 获取当前状态下需要采取的动作
                action = agent.take_action(state)
                action_list.append(action)
                # 更新环境
                next_state, reward, done = env.step(action,action_list)
                # 添加经验池
                replay_buffer.add(state, action, reward, next_state, done)
                # 更新当前状态
                state = next_state
                # 更新回合回报
                episode_return += reward

                # 当经验池超过一定数量后，训练网络
                if replay_buffer.size() > min_size:
                    # 从经验池中随机抽样作为训练集
                    s, a, r, ns, d = replay_buffer.sample(batch_size)
                    # 构造训练集
                    transition_dict = {
                        'states': s,
                        'actions': a,
                        'next_states': ns,
                        'rewards': r,
                        'dones': d,
                    }
                    # 网络更新
                    loss.append(agent.update(transition_dict))
                # 找到目标就结束
                if done: break
            # 记录每个回合的回报
            return_list.append(episode_return)

            # 更新进度条信息
            pbar.set_description(desc= "Iteration %d" %i)
            pbar.set_postfix({
                'return': '%.3f' % np.mean(return_list),
                "loss": '%f' % np.mean(loss) if len(loss) > 0 else 0
            })
            pbar.update(1)

    loss_list.append(np.mean(loss))

    #贪心系数衰减
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

#保存模型
agent.model_save()

# 绘图
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN Returns')
plt.show()
