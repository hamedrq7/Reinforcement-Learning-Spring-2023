import gym
import numpy as np
from typing import Dict, List, Tuple
import math
import matplotlib.pyplot as plt

def print_v(value_func):
    for i in range(4):
        print(value_func[i*4 : (i+1)*4])

def test_policy(env, policy, gamma, iteration, is_VI):
    n_eps = 1000
    n_steps = 1000
    _, _, avg_return, avg_steps = run_game(env, n_eps, policy, max_num_steps=n_steps)
    print('-'*20)
    if is_VI:
        print('VI - iteration ', iteration, ' gamma ', gamma)
    else:
        print('PI - iteration ', iteration, ' gamma ', gamma)
        
    print('avg_return: ', avg_return)
    print('avg_steps: ', avg_steps)


def plot_policy(policy, env):  
    
    # fig, ax = plt.subplots()
    # # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    # ax.matshow(data, cmap='seismic')

    # for (i, j), z in np.ndenumerate(data):
    #     ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
    #             bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    # plt.show()

    print(policy)    
    arrows = []
    # 0 left
    # 1 down
    # 2 right
    # 3 up
    for i in range(4): 
        arrow_row = []
        for j in range(4):
            # i*4 + j
            scales = []
            for idx, action in enumerate(policy[i*4 + j].keys()):
                arr_size = 0.0
                if policy[i*4 + j][action] > 0.0: 
                    arr_size = 0.25
                # scales.append(policy[i*4 + j][action] * 0.4) 
                scales.append(arr_size)
            arrow_row.append(scales)
        arrows.append(arrow_row)

    action_dict = {2:(1,0), 0:(-1,0),3:(0,1),1:(0,-1)}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0.5, 4.5)
    ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=5))
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=5))
    
    for r, row in enumerate(arrows):
        for c, cell in enumerate(row):
            for temp in range(4):
                ax.arrow(c, 4-r, 
                          dx=arrows[r][c][temp]*action_dict[temp][0], # ha='center', va='center', 
                          dy=arrows[r][c][temp]*action_dict[temp][1], head_width=0.2*arrows[r][c][temp])

                ax.text(c, 4-r, '{}'.format(env.desc[r][c].decode('UTF-8')),  ha='center', va='center',
                      bbox=dict(boxstyle='round', facecolor=get_box_color(env.desc[r][c].decode('UTF-8')), edgecolor='0.3')
                     )


    ax.grid(lw=5.0)
    plt.show()

       
def get_box_color(state: str):
    if state == 'S':
        return 'lightblue'
    elif state == 'F':
        return 'white'
    elif state == 'H':
        return 'red'
    elif state == 'G':
        return 'lightgreen'
    else: 
        print('invalid state')
        exit()

def get_init_policy(nS=16) -> Dict[int, Dict[int, float]]:
    policy = {}
    sub_dict = {
        0: 0.25,
        1: 0.25,
        2: 0.25,
        3: 0.25
    } 
    for state in range(nS):
        policy[state] = sub_dict
    
    return policy

def OptActions(policy: Dict[int, Dict[int, float]], state):
    act = np.random.choice(np.array(list(policy[state].keys())), size=1, p=np.array(list(policy[state].values())))[0]
    return act

def plot_value_func_policy(env, policy, value_function, name_to_save: str = '', title: str = ''):
    value_function = value_function.reshape((4, 4))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax0.matshow(value_function, cmap='seismic')

    for (i, j), z in np.ndenumerate(value_function):
        ax0.text(j, i, '{:0.6f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    arrows = []
    # 0 left
    # 1 down
    # 2 right
    # 3 up
    for i in range(4): 
        arrow_row = []
        for j in range(4):
            # i*4 + j
            scales = []
            for idx, action in enumerate(policy[i*4 + j].keys()):
                arr_size = 0.0
                if policy[i*4 + j][action] > 0.0: 
                    arr_size = 0.25
                # scales.append(policy[i*4 + j][action] * 0.4) 
                scales.append(arr_size)
            arrow_row.append(scales)
        arrows.append(arrow_row)

    action_dict = {2:(1,0), 0:(-1,0),3:(0,1),1:(0,-1)}

    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(0.5, 4.5)
    ax1.set_xticks(np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], num=5))
    ax1.set_yticks(np.linspace(ax1.get_ylim()[0], ax1.get_ylim()[1], num=5))
    
    for r, row in enumerate(arrows):
        for c, cell in enumerate(row):
            for temp in range(4):
                ax1.arrow(c, 4-r, 
                          dx=arrows[r][c][temp]*action_dict[temp][0], # ha='center', va='center', 
                          dy=arrows[r][c][temp]*action_dict[temp][1], head_width=0.2*arrows[r][c][temp])

                ax1.text(c, 4-r, '{}'.format(env.desc[r][c].decode('UTF-8')),  ha='center', va='center',
                      bbox=dict(boxstyle='round', facecolor=get_box_color(env.desc[r][c].decode('UTF-8')), edgecolor='0.3')
                     )


    ax1.grid(lw=5.0)
    
    if name_to_save == '':
        plt.show()
    else: 
        plt.title(title)
        plt.savefig(f'imgs/{name_to_save}.png')

def run_game(env, num_episodes: int, policy: Dict[int, Dict[int, float]], 
             max_num_steps: int = 50000) -> Tuple[np.ndarray, np.ndarray, np.number, np.number]:
    return_hist = np.zeros((num_episodes))
    steps_hist  = np.zeros((num_episodes))

    env.reset()
    for curr_ep in range(num_episodes):
        curr_state = 0
        
        for steps in range(max_num_steps):
            action = OptActions(policy, curr_state)
            observation, reward, terminated, truncated, info = env.step(action)

            return_hist[curr_ep] += reward
            curr_state = observation
            if truncated or terminated:
                break
        env.reset()

        steps_hist[curr_ep] = steps+1

    avg_return = np.mean(return_hist)
    avg_steps  = np.mean(steps_hist)
    return return_hist, steps_hist, avg_return, avg_steps

def plot_value_function(data):
    data = data.reshape((4, 4))
    fig, ax = plt.subplots()
    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.matshow(data, cmap='seismic')

    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{:0.6f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.show()

def update_policy_greedy(P, value_function: np.ndarray, gamma) -> Dict[int, Dict[int, float]]:
    # update policy: 
    
    new_policy = {}
    for curr_state in P.keys():
        
        temp_values = np.zeros(len(list(P[curr_state].keys())))

        for curr_action in P[curr_state].keys():
            for idx, transition in enumerate(P[curr_state][curr_action]):
                (prob_of_transition, s_prime, reward, terminated) = transition
                
                temp_values[curr_action] += prob_of_transition * (reward + gamma * value_function[s_prime])

        winners = np.argwhere(temp_values == np.max(temp_values))[:, 0]
        num_winners = winners.shape[0]
        # equal prob for actions with best_action_value
        new_policy_actions = {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.0,
        }
        for winner in winners: 
            new_policy_actions[winner] = 1.0 / num_winners
        new_policy[curr_state] = new_policy_actions
    
    return new_policy

def check_equal_policy(policy1, policy2):
    if policy1 == policy2:
        return True
    return False
