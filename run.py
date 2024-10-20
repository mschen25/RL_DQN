import pandas as pd
from Environment import Environment
from RL_DQN import DQN
import torch
from tqdm import tqdm
import numpy as np
import time
from MDP_model import MDP

# cpu运算
device = torch.device("cpu")



n_hidden = 96  # 隐含层神经元个数
n_states = 5  # 状态个数
n_actions = 100 #动作个数
Is_train = True #是否有模型

result_list = [] #汇总结果记录
df = []
#加载环境
env = Environment()

#获取电动车数
mdp= MDP()
num = mdp.get_len()

#计算程序运行时间
start_time = time.time()

agent = DQN(n_states=n_states,
            n_hidden=n_hidden,
            n_actions=n_actions,

            device=device,
            Is_train=Is_train
            )
with tqdm(total=num) as pbar:
    for i in range(num):
        path=[] #路径储存列表
        E_and_T_list=[] #电量和时间储存列表
        EVs = env.make(i) #储存电动车信息
        EVs.insert(0,i)
        state = env.reset()

        # 记录经过的点
        action_list = [state[0]]

        while True:
            #获取动作
            action = agent.take_action(state,action_list)
            #添加动作到已执行动作表
            action_list.append(action)
            #更新状态
            next_state, reward, done =env.step(action)
            #储存电量和时间
            E_and_T_list.append([next_state[2], next_state[1]])
            if not done:
                #储存路径
                path.append( (int(state[0]), int(next_state[0])) )

                # 更新当前状态
                state = next_state
            else:
                path.append( (int(state[0]),int(next_state[0])))
                break

        # 对到达终点的车进行记录
        if E_and_T_list[-1][1] >= 0 and E_and_T_list[-1][0] > 0 :
            df.append(EVs)
            df.append(path)
            df.append(E_and_T_list)

            result_list.append([i]+E_and_T_list[-1])

        pbar.update(1)

#结束时间
end_time = time.time()
Time = end_time - start_time

#计算总电量
total = np.sum(result_list,axis=0)[1]
#添加总电量
result_list.append(["total",total])
#把列表数据转换成数据框
dataFrame1= pd.DataFrame(data=df,index=None)
dataFrame2 = pd.DataFrame(data=result_list,index=None)

with pd.ExcelWriter('../输出结果/results.xlsx') as writer:
    dataFrame1.to_excel(excel_writer=writer,index=False,header=False,sheet_name="调度详情")
    dataFrame2.to_excel(excel_writer=writer,index=False,header=False,sheet_name="汇总")

print("总电量："+str(total))
print("运行时间："+str(Time))