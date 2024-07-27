

from env import Env



env=Env()

#调度5===========================================调度6==================================
# env.step(122)3                            111，112，111，122，112，112，142，212,142,112
# env.step(212)7
# env.step(111)0

# env.step(122)3
# env.step(111)0
# env.step(122)3

# env.step(122)3
# env.step(111)0
# env.step(122)3
#==================================================================================
episode_return = 0
done = False
return_list = []  # 保存每个回合的return
# 构造数据集，保存每个回合的状态数据
transition_dict = {
    'states': [],
    'actions': [],
    'next_states': [],
    'rewards': [],
    'dones': [],
}
state =env.INITSTATE

i=0
while not done:
    #action=[3,7,0,3,0,3,3,0,3]  # 动作选择
    action=[0,1,0,3,1,1, 19,7,19,1]

    next_state, reward, done = env.step(action[i],True,3)  # 环境更新

    # 保存每个时刻的状态\动作\...
    transition_dict['states'].append(state)
    transition_dict['actions'].append(action[i])
    transition_dict['next_states'].append(next_state)
    transition_dict['rewards'].append(reward)
    transition_dict['dones'].append(done)
    # 更新状态
    state = next_state
    # 累计回合奖励
    episode_return += reward
    i += 1

# 保存每个回合的return
return_list.append(episode_return)


print("The return is:",episode_return)



