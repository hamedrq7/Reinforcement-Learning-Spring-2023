import gym
import numpy as np
from utils import * 
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import math


def value_iteration(env, P, nS, nA, gamma=0.9, tol=1e-4):
    '''
    parameters:
        P: transition probability matrix
        nS: number of states
        nA: number of actions
        gamma: discount factor
        tol: tolerance for convergence
    returns:
        value_function: value function for each state
        policy: policy for each state
    '''
    # initialize value function and policy
    value_function = np.zeros(nS)

    policy = get_init_policy()

    iter_idx = 0
    delta = tol + 1
    while delta > tol:
        iter_idx += 1
        delta = 0.0

        for curr_state in P.keys():
            old_state_value = value_function[curr_state]

            new_max_state_value = -math.inf
            for curr_action in P[curr_state].keys():
                curr_action_value = 0.0
                for idx, transition in enumerate(P[curr_state][curr_action]):
                    (prob_of_transition, s_prime, reward, terminated) = transition
                    curr_action_value += prob_of_transition * (reward + gamma * value_function[s_prime])
                
                new_max_state_value = max(new_max_state_value, curr_action_value)

            value_function[curr_state] = new_max_state_value
            delta = max(delta, np.abs(value_function[curr_state] - old_state_value))
        
        policy = update_policy_greedy(P, value_function, gamma)
        test_policy(env, policy, gamma, iter_idx, True)


    # update policy
    policy = update_policy_greedy(P, value_function, gamma)
    
    return value_function, policy


# if __name__ == "__main__":

#     env = gym.make("FrozenLake-v1", is_slippery=True, render_mode='ansi')
    
#     env.reset()
    
#     value_function, policy = value_iteration(env.P, env.observation_space.n, env.action_space.n, gamma=0.9, tol=1e-4)

#     _, _, avg_return , avg_steps = run_game(env, 1000, policy)
#     print(avg_return, avg_steps)
#     plot_value_func_policy(env, policy, value_function)
#     env.close()