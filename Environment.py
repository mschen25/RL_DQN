from MDP_model import MDP
import numpy as np

class Environment:
    def __init__(self):
        # 初始状态
        self.ori_state=None
        # 下一个state
        self.next_state=None
        # 此前的奖励积累
        self.reward=None
        # 此前的运行状态 true为完成
        self.done=False
        #终点
        self.end = None
        self.mdp = MDP()

    # 返回初始状态
    def reset(self):
        return np.array(self.ori_state)

    # 更新初始状态
    def make(self,i):
        self.ori_state=self.mdp.set_state_and_target(i)#更新ori_state
        self.end = self.mdp.get_end_point()

        #返回电动车数据
        return list(self.mdp.get_evs(i))

    def step(self,action, action_list = []):
        # 更新当前状态和下一状态
        state, self.next_state=self.mdp.state_transfer(action)

        # 更新运行状态done
        if self.next_state[1] < 0 or self.next_state[0] == self.end or self.next_state[2] <=0:
            self.done=True
        else:
            self.done=False
        # 更新奖励
        self.reward = self.mdp.get_reword(state,self.next_state, action_list, self.done)

        return np.array(self.next_state), self.reward, self.done


if __name__ == "__main__" :
    env=Environment()
    env.make(1)
    a = env.reset()
    ns,r,d = env.step(26)
