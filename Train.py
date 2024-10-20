import pandas as pd

from Environment import Environment
from RL_DQN import DQN, ReplayBuffer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from MDP_model import MDP
# GPU运算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------- #
# 全局变量
# ------------------------------- #
important_capacity = 15000 #重要经验池
capacity = 45000  # 经验池容量
lr = 2e-4  # 学习率
gamma = 0.95  # 折扣因子
epsilon = 0.5 # 贪心系数 小于探索
epsilon_min = 0.05 #最小贪心系数
epsilon_decay = 0.98 #贪心系数折扣率
target_update = 1024  # 目标网络的参数的更新频率
batch_size = 1024  # 更新网络的数据量
n_hidden = 96  # 隐含层神经元个数
min_size = 1024  # 普通经验池超过多少后再训练
return_list = []  # 记录每个回合的回报

loss = [] # 记录每回合每次更新网络产生是loss
loss_list = [] #用来保留每回合平均loss
Is_train = True  #是否有训练模型
episode = 200

# 每几回合保存一次网络参数
save_model = 50

# 加载环境

env = Environment()
n_states = 5
n_actions = 100
# 获取电动车个数
mdp= MDP()
num = mdp.get_len()
reward_return = np.array([[x] for x in range(num)]) #记录所有回报

# 实例化经验池
replay_buffer = ReplayBuffer(capacity,important_capacity)
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
    with tqdm(total=num) as pbar:
        # 训练模型
        for Id in range(num):
            # 每个回合开始前重置环境
            env.make(Id)
            # 记录初始状态
            state = env.reset()
            # 记录每个回合的回报
            episode_return = 0
            # 每次动作路径
            action_list = [state[0]]
            done = False #是否结束

            while True:
                # 获取当前状态下需要采取的动作
                action = agent.take_action(state,action_list)

                # 更新环境
                next_state, reward, done = env.step(action,action_list)
                # 添加动作
                action_list.append(action)
                # 添加经验池
                if reward < 1:
                    replay_buffer.add(state, action, reward, next_state, done)
                else:
                    replay_buffer.add_important(state, action, reward, next_state, done)
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
    reward_return = np.column_stack((reward_return, return_list))
    #贪心系数衰减
    if epsilon > epsilon_min and (i+1) % 5 == 0:
        epsilon *= epsilon_decay
        agent.update_epsilon(epsilon)

    if (i+1) % save_model == 0 :
        #保存模型
        agent.model_save(i+1)

df_reward = pd.DataFrame(reward_return, index=None)
df_reward.to_csv("../输出结果/rewards.csv", header=False, index=False)

# 绘图
episodes_list = list(range(episode))

plt.figure(1)
plt.plot(episodes_list, loss_list)
plt.xlabel('training steps')
plt.ylabel('Cost')
plt.savefig("../输出结果/training_cost.jpg")

reward_mean = np.mean(reward_return,axis=0)

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 显示负号
plt.rcParams['axes.unicode_minus'] = False

plt.figure(2)
plt.plot(episodes_list, reward_mean[1:])
plt.xlabel("慕(训练次数)")
plt.ylabel("平均奖励")
plt.savefig("../输出结果/episode_rewards.jpg")

plt.show()