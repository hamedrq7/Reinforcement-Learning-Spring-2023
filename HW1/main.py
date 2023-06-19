import gym
import numpy as np
from gym.wrappers import transform_reward
from policy_iteration import policy_iteration
from value_iteration import value_iteration
from utils import * 

def change_of_gamma():
    # policy iteration
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode='ansi')
    env.reset()
    
    for gamma in [0, 0.1, 0.9, 1]:    
        value_function, policy = policy_iteration(env, env.P, env.observation_space.n, env.action_space.n, gamma=gamma, tol=1e-4)

        
        plot_value_func_policy(env, policy, value_function, f'change_of_gamma_{gamma} policy_iteration', f'change_of_gamma_{gamma} policy_iteration')
        
        env.render()
        env.close()

    # value iteration
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode='ansi')
    env.reset()
    for gamma in [0, 0.1, 0.9, 1]:    
        value_function, policy = value_iteration(env, env.P, env.observation_space.n, env.action_space.n, gamma=gamma, tol=1e-4)

        # _, _, avg_return , avg_steps = run_game(env, 1000, policy)
        # print(avg_return, avg_steps)
        plot_value_func_policy(env, policy, value_function, f'change_of_gamma_{gamma} value_iteration', f'change_of_gamma_{gamma} value_iteration')
        env.close()

def non_deterministic():
    # policy iteration
    # policy iteration
    nondeterministic = True
    env = gym.make("FrozenLake-v1", is_slippery=nondeterministic, render_mode='ansi')
    env.reset()
    
    gamma = 0.9
    value_function, policy = policy_iteration(env, env.P, env.observation_space.n, env.action_space.n, gamma=gamma, tol=1e-4)

    plot_value_func_policy(env, policy, value_function, f'nondeterminisitc policy_iteration, gamma {gamma}', f'nondeterminisitc policy_iteration, gamma {gamma}')
    
    env.render()
    env.close()
    
    # value iteration
    nondeterministic = True
    env = gym.make("FrozenLake-v1", is_slippery=nondeterministic, render_mode='ansi')
    env.reset()
    
    gamma = 0.9
    value_function, policy = value_iteration(env, env.P, env.observation_space.n, env.action_space.n, gamma=gamma, tol=1e-4)

    plot_value_func_policy(env, policy, value_function, f'nondeterminisitc value_iteration, gamma {gamma}', f'nondeterminisitc value_iteration, gamma {gamma}')
    
    env.render()
    env.close()



class MovePenaltyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_state = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if observation != self.prev_state:
            self.prev_state = observation
            return observation, -0.05, terminated, truncated, info
        else:
            self.prev_state = observation
            return observation, reward, terminated, truncated, info
      
def move_penalty():
    # policy iteration
    nondeterministic = False
    env = MovePenaltyWrapper(gym.make("FrozenLake-v1", is_slippery=nondeterministic, render_mode='ansi'))
    env.reset()
    
    gamma = 0.9
    value_function, policy = policy_iteration(env, env.P, env.observation_space.n, env.action_space.n, gamma=gamma, tol=1e-4)

    # plot_value_func_policy(env, policy, value_function, f'move -0.05 penalty policy_iteration, gamma {gamma}', f'move -0.05 penalty policy_iteration, gamma {gamma}')
   
    env.render()
    env.close()
    
    # value iteration
    nondeterministic = False
    env = MovePenaltyWrapper(gym.make("FrozenLake-v1", is_slippery=nondeterministic, render_mode='ansi'))
    env.reset()
    
    gamma = 0.9
    value_function, policy = value_iteration(env, env.P, env.observation_space.n, env.action_space.n, gamma=gamma, tol=1e-4)
    
    # plot_value_func_policy(env, policy, value_function, f'move -0.05 penalty value_iteration, gamma {gamma}', f'move -0.05 penalty value_iteration, gamma {gamma}')
    
    env.render()
    env.close()



class HolePenaltyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.Holes = [5, 7, 11, 12]

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if observation in self.Holes:
            self.prev_state = observation
            return observation, -2, terminated, truncated, info
        else:
            return observation, reward, terminated, truncated, info
        
def hole_penalty():
    # policy iteration
    nondeterministic = False
    env = HolePenaltyWrapper(gym.make("FrozenLake-v1", is_slippery=nondeterministic, render_mode='ansi'))
    env.reset()
    
    gamma = 0.9
    value_function, policy = policy_iteration(env, env.P, env.observation_space.n, env.action_space.n, gamma=gamma, tol=1e-4)

    plot_value_func_policy(env, policy, value_function, f'Hole -2 penalty policy_iteration, gamma {gamma}', f'Hole -2 penalty policy_iteration, gamma {gamma}')
    _, _, a, b = run_game(env, 1000, policy)
    print(a, b)
    env.render()
    env.close()
    
    # value iteration
    nondeterministic = False
    env = HolePenaltyWrapper(gym.make("FrozenLake-v1", is_slippery=nondeterministic, render_mode='ansi'))
    env.reset()
    
    gamma = 0.9
    value_function, policy = value_iteration(env, env.P, env.observation_space.n, env.action_space.n, gamma=gamma, tol=1e-4)
    
    plot_value_func_policy(env, policy, value_function, f'Hole -2 penalty value_iteration, gamma {gamma}', f'Hole -2 penalty value_iteration, gamma {gamma}')
    
    env.render()
    env.close()



# change_of_gamma()
# non_deterministic()

# move_penalty()
hole_penalty()
