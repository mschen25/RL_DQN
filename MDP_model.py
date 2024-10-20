
import pandas as pd
import numpy as np


class MDP:
    def __init__(self):
        #点的经纬度
        self.data = np.array(pd.read_csv("./数据集/data.csv", header=None))
        #各点之间的距离
        self.distance = np.array(pd.read_csv("./数据集/distance.csv", header=None))
        #该点是否能充电
        self.charge_roads = np.array(pd.read_csv("./数据集/roads.csv", header=None))
        #在该路径的速度
        self.speed = np.array(pd.read_csv("./数据集/speed.csv", header=None))
        #4-7列分别为初始电量，满电量，剩余时间，行驶耗能
        self.EVs_50 = np.array(pd.read_csv("./数据集/EVs.csv", header=None))
        self.MPT = 100  # 充电功率

        #把EVs_50中的经纬度换成对应点
        point = []
        for i in self.EVs_50[:,:2]:
            point.append(np.where(self.data == i)[0][0])
        self.EVs = np.array(point).reshape(-1,1)
        point = []
        for i in self.EVs_50[:,2:4]:
            point.append(np.where(self.data == i)[0][0])
        point = np.array(point).reshape(-1,1)
        self.EVs = np.append(self.EVs, values=point, axis=1)
        # 0-5列分别为起点，终点，初始电量，满电量，剩余时间，行驶耗能
        self.EVs = np.append(self.EVs, values=self.EVs_50[:,4:8], axis=1)

        self.E_max = None  # 最大电量
        self.E = None  # 初始电量
        self.cost = None # 行驶耗能
        self.time = None # 截止时间
        self.end =None  #终点
        self.start = None #起点
        self.state = [] #状态列表
        #记录已获取的满电奖励次数
        self.E_max_num = 0
    #根据电动车类型设置初始状态和目标
    def set_state_and_target(self,i):
        self.E_max = self.EVs[i][3]
        self.E = self.EVs[i][2]
        self.time = self.EVs[i][4]
        self.cost = self.EVs[i][5]/100.0
        self.end = self.EVs[i][1]
        self.start = int(self.EVs[i][0])
        self.state = [self.start, self.time, self.E, self.end, self.E_max]
        # 初始化满电奖励次数
        self.E_max_num = 0
        return self.state

    def get_reword(self, state, next_state, action_list, done):
        r = 0
        # 到达终点且未超时
        if next_state[0] == self.end and next_state[1] >= 0 and done:
            r += 1
            # 根据电量给额外奖励
            r += next_state[2] / self.E_max

        # 未到终点 或 电量为0 或 时间为0
        if (next_state[0] != self.end or next_state[1] < 0 or next_state[2] <= 0) and done:
            r += -1

        #状态不变
        if state[1] == next_state[1]:
            r += -0.8

        #满电量获取次数不超过2次
        # if next_state[2] == self.E_max and state[1] != next_state[1] and self.E_max_num < 2:
        #     r += 0.9
        #     self.E_max_num += 1

        #电量变化
        if next_state[2] - state[2] > 0:
            r += (next_state[2] - state[2]) / self.E_max
        else:
            r += (next_state[2] - state[2]) / (self.cost*100)

        return r

    #返回当前状态和下一状态
    def state_transfer(self,action):
        #是否充电
        a =self.charge_roads[self.state[0]][action]

        distance = self.distance[self.state[0]][action]
        speed = self.speed[self.state[0]][action]

        # 时间变化
        diet_t = distance/(speed + 1e-8)

        #电量变化
        e = self.state[2] - self.cost*distance + a*self.MPT*diet_t
        E = min(e, self.E_max)
        t = self.state[1] - diet_t

        #更新状态
        next_state = [action, t, E, self.end, self.E_max]
        state = self.state
        self.state = next_state

        return state, next_state

    def get_end_point(self):
        return self.end

    def get_evs(self,i):
        return self.EVs[i,:5]

    def get_len(self):
        return len(self.EVs)