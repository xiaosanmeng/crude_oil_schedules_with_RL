import copy

from env import Env



env=Env()

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

    # action= [9,7,9,20, 0, 2, 0,1]#[20, 10, 13, 10, 238.0]
    # action=[9,7,9,14,0,14,0,13]

    action=[7,9,9,11,0,5,0,5]
    next_state, reward, done = env.step(action[i],True,-1)

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
print("The object is:",env.preTarget)



