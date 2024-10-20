import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
import random



# --------------------------------------- #
# 经验回放池
# --------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity, important_capacity):
        # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
        self.buffer = collections.deque(maxlen=capacity)
        # 创建一个重要经验池
        self.important_buffer = collections.deque(maxlen=important_capacity)
        self.now_experience =None
        self.important_scale = int(capacity / important_capacity)

    # 将数据以元组形式添加进经验池
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.now_experience = (state, action, reward, next_state, done)

    def add_important(self, state, action, reward, next_state, done):
        self.important_buffer.append((state, action, reward, next_state, done))
        self.now_experience = (state, action, reward, next_state, done)

    # 随机采样batch_size行数据
    def sample(self, batch_size):
        important_size = min(int(batch_size / self.important_scale), len(self.important_buffer))
        common_size = batch_size - important_size - 1
        transitions = random.sample(self.important_buffer, important_size)
        transitions.extend(random.sample(self.buffer, common_size))
        transitions.append(self.now_experience)
        # *transitions代表取出列表中的值，即32项
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    # 目前队列长度
    def size(self):
        return len(self.buffer)


# -------------------------------------- #
# 构造深度学习网络，输入状态s，得到各个动作的reward
# -------------------------------------- #

class Net(nn.Module):
    # 构造只有一个隐含层的网络
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        # [b,n_states]-->[b,n_hidden]
        self.fc1 = nn.Linear(n_states, n_hidden)

        self.fh1 = nn.Linear(n_hidden, n_hidden)
        self.fh2 = nn.Linear(n_hidden, n_hidden)
        self.fh3 = nn.Linear(n_hidden, n_hidden)
        self.fh4 = nn.Linear(n_hidden, n_hidden)
        self.fh5 = nn.Linear(n_hidden, n_hidden)

        # [b,n_hidden]-->[b,n_actions]
        self.fc2 = nn.Linear(n_hidden, n_actions)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
    # 前传
    # 激活函数为ReLU
    def forward(self, x):  # [b,n_states]
        x = self.leaky_relu(self.fc1(x))

        x = self.leaky_relu(self.fh1(x))
        x = self.leaky_relu(self.fh2(x))
        x = self.leaky_relu(self.fh3(x))
        x = self.leaky_relu(self.fh4(x))
        x = self.leaky_relu(self.fh5(x))

        x = self.fc2(x)

        return x


# -------------------------------------- #
# 构造深度强化学习模型
# -------------------------------------- #

class DQN:
    # （1）初始化
    def __init__(self, n_states, n_hidden, n_actions,
                 learning_rate=1e-3 , gamma=0, epsilon=0,
                 target_update=0, device=None,Is_train=False):
        # 属性分配
        self.n_states = n_states  # 状态的特征数
        self.n_hidden = n_hidden  # 隐含层个数
        self.n_actions = n_actions  # 动作数
        self.learning_rate = learning_rate  # 训练时的学习率
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device  # 在GPU计算
        # 计数器，记录迭代次数
        self.count = 0
        self.Is_train = Is_train

        # 状态最大张量
        self.state_max = torch.tensor([100, 5, 100, 100, 100]).to(self.device)
        # 状态最小张量
        self.state_min = torch.tensor([0, 0, 0, 0, 40]).to(self.device)

        # 构建2个神经网络，相同的结构，不同的参数
        # 实例化训练网络  [b,4]-->[b,50]  输出动作对应的奖励
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions).to(self.device)
        # 实例化目标网络
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        #加载网络模型参数
        if self.Is_train :
            self.q_net.load_state_dict(torch.load("q_net.pth", weights_only=True))
            self.target_q_net.load_state_dict(torch.load("q_net.pth", weights_only=True))
        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    # 状态最大最小归一化
    def min_max_normalize(self, state):
        return (state - self.state_min) / (self.state_max - self.state_min)

    # （2）动作选择
    def take_action(self, state,action_list):
        # 维度扩充，给行增加一个维度，并转换为张量shape=[1,4]
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
        # 如果大于该值就取最大的值对应的索引
        if np.random.random() >= self.epsilon:  # 0-1
            # 前向传播获取该状态对应的动作的reward
            actions_value = self.q_net(self.min_max_normalize(state))
            # 根据动作列表和Q值 选取未进行过的动作
            while True:
                # 获取reward最大值对应的动作索引
                action = actions_value.argmax().item()  # int
                if action not in action_list:
                    break
                else:
                    actions_value.data[0,action] = -float('inf')
        # 如果小于该值就随机探索
        else:
            while True:
                # 随机选择一个动作
                action = np.random.randint(self.n_actions)
                if action not in action_list:
                    break

        return action

    # （3）网络训练
    def update(self, transition_dict):  # 传入经验池中的batch个样本
        # 获取当前时刻的状态 array_shape=[b,5]
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # 获取当前时刻采取的动作 tuple_shape=[b]，维度扩充 [b,1]
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        # 当前状态下采取动作后得到的奖励 tuple=[b]，维度扩充 [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        # 下一时刻的状态 array_shape=[b,5]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        # 是否到达目标 tuple_shape=[b]，维度变换[b,1]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 输入当前状态，得到采取各运动得到的奖励 [b,5]==>[b,50]==>[b,1]
        # 根据actions索引在训练网络的输出的第1维度上获取对应索引的q值（state_value）
        q_values = self.q_net(self.min_max_normalize(states)).gather(1, actions)  # [b,1]

        # 下一时刻的状态[b,5]-->目标网络输出下一时刻对应的动作q值[b,50]-->
        # 选出下个状态采取的动作中最大的q值[b]-->维度调整[b,1]
        max_next_q_values = self.target_q_net(self.min_max_normalize(next_states)).max(1)[0].view(-1, 1)
        # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 目标网络和训练网络之间的均方误差损失
        dqn_loss = F.mse_loss(q_values, q_targets)
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        # 反向传播参数更新
        dqn_loss.backward()
        # 对训练网络更新
        self.optimizer.step()

        self.count += 1

        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())

        return dqn_loss.item()



    # (4)模型保存
    def model_save(self,num):
        torch.save(self.q_net.state_dict(),'q_net%d.pth' %num)
        torch.save(self.target_q_net.state_dict(),'target_net%d.pth' %num)

    def update_epsilon(self,epsilon):
        self.epsilon = epsilon

    # 通过综合两个网络来选取动作
    def take_action_with_two_net(self, state, action_list):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
        actions_value_q_net = self.q_net(self.min_max_normalize(state))
        actions_value_target_net = self.target_q_net(self.min_max_normalize(state))
        while True:
            a = actions_value_q_net.max()
            b= actions_value_target_net.max()
            if a > b:
                action = actions_value_q_net.argmax().item()
                if action in action_list:
                    actions_value_q_net.data[0, action] = -float('inf')
                else:
                    return action
            else:
                action = actions_value_target_net.argmax().item()
                if action in action_list:
                    actions_value_target_net.data[0, action] = -float('inf')
                else:
                    return action