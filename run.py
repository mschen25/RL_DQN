from Environment import Environment
from RL_DQN import DQN
import torch
from tqdm import tqdm
import numpy as np


# GPU运算
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

capacity = 500  # 经验池容量
lr = 1e-3  # 学习率
gamma = 0.98  # 折扣因子
epsilon = 0.4  # 贪心系数
epsilon_min = 0.1 #最小贪心系数
epsilon_decay = 0.98 #贪心系数折扣率
target_update = 50  # 目标网络的参数的更新频率
batch_size = 32
min_size = 200  # 经验池超过多少后再训练
episode = 1000

n_hidden = 64  # 隐含层神经元个数
n_states = 4  # 状态个数
n_actions = 50 #动作个数
Is_train = True #是否有模型
result_list = [] #汇总结果记录
#加载环境
env = Environment()

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
with tqdm(total=50) as pbar:
    for i in range(50):
        path=[] #路径储存列表
        E_and_T_list=[] #电量和时间储存列表
        EVs = env.make(i)
        EVs.insert(0,i)
        state = env.reset()

        while True:
            #获取动作
            action = agent.take_action(state)
            #更新状态
            next_state, reward, done =env.step(action)
            if not done:
                #储存路径
                path.append( (int(state[0]), int(next_state[0])) )
                #储存电量和时间
                E_and_T_list.append([next_state[2], next_state[1]])
                # 更新当前状态
                state = next_state
            else:
                path.append( (int(state[0]),int(state[3])))
                break
        result_list.append(E_and_T_list[-1])
        pbar.update(1)

total = sum(result_list[:][0])
print(total)