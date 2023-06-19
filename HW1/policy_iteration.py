import gym
import numpy as np
from typing import Dict, List, Tuple
import math
from utils import * 


def policy_evaluation(value_function: np.ndarray, 
                      P: Dict[int, Dict[int, List]], # Dict[int(state), Dict[int(action), List[prob of action: float, next_state: int, reward: float]]]
                      policy: Dict[int, Dict[int, float]], # Dict[int(state), Dict[int(action), prob: float]] 
                      nS: int, nA: int, gamma: float=0.9, tol: float=1e-4):
    max_iter = 1e6
    iter = 0
    delta = tol + 1
    
    while delta > tol:

        iter += 1
        if iter == max_iter: 
            break

        delta = 0.0
        for curr_state in range(nS):
            old_state_value_func = value_function[curr_state]
            # P[curr_state] -> dict = {
            #   action_0: List[prob_action_0, prob_action_1, prob_action_2, LeadsToTermination],
            #   action_1: List[prob_action_0, prob_action_1, prob_action_2, LeadsToTermination],
            #   ...
            # }
            new_state_value_func = 0.0
            # print(curr_state)
            for curr_action in P[curr_state].keys():
                prob_of_selecting_action = policy[curr_state][curr_action]
                    
                for idx, transition in enumerate(P[curr_state][curr_action]):
                    (prob_of_transition, s_prime, reward, terminated) = transition
                    # print(prob_of_transition, s_prime, reward, terminated, '<- prob trans, next state, reward, termination')

                    new_state_value_func += prob_of_selecting_action * prob_of_transition * (reward + gamma * value_function[s_prime])

            value_function[curr_state] = new_state_value_func

            delta = max(delta, np.abs(value_function[curr_state] - old_state_value_func))


        # plot_value_function(value_function)
    
    # print('number of iterations in evaluation: ', iter)
    return value_function
        

def policy_iteration(env, P, nS, nA, gamma=0.9, tol=1e-4):
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
    # policy:  Dict[int(state), Dict[int(action), prob_of_selecting: float]] 
    policy = get_init_policy()

    iter_idx = 0
    while True:
        iter_idx += 1
        value_function = policy_evaluation(value_function, P, policy, nS, nA, gamma, tol)

        # print_v(value_function)
        
        # policy improvement
        new_policy = update_policy_greedy(P, value_function, gamma)
        # print(policy)

        # plot_value_func_policy(env, new_policy, value_function)
        # _, _, aa, aaa = run_game(env, 1000, policy)
        # print(aa, aaa)
        # exit()
        
        if check_equal_policy(new_policy, policy) or iter_idx == 10000:
            print('out of loop')
            break

        policy = new_policy

        # test policy: 
        test_policy(env, policy, gamma, iter_idx, False)


    return value_function, policy


# if __name__ == "__main__":
    
#     DESC = ["SFFFFG"] 
#     env = gym.make("FrozenLake-v1", is_slippery=True, render_mode='ansi')
#     env.reset()
    
#     value_function, policy = policy_iteration(env.P, env.observation_space.n, env.action_space.n, gamma=0.9, tol=1e-4)

#     return_hist, steps_hist, avg_return, avg_steps = run_game(env, 100, policy)
#     print(avg_return, avg_steps)
#     print(policy)
#     print_v(value_function)
#     plot_value_func_policy(env, policy, value_function)
    
#     env.render()
#     env.close()