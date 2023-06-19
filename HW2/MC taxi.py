import gym
from utils import * 

# define parameters: 
alpha = 0.1  # Learning rate
epsilon = 0.75  # Exploration rate 
num_episodes = 10000  # Number of episodes

for gamma in [0.9, 0.0, 0.5, 0.01]: # Discount rate
    print('running MC experiment ', alpha, epsilon, gamma, num_episodes)
    
    env = gym.make("Taxi-v3", render_mode="ansi")
    # env = gym.make("Taxi-v3", render_mode="ansi")
    env = BasicWrapper(env) # wrapper for 4 more movements 

    policy = eps_greedy_policy(env, epsilon, is_mc=True)
    agent = RL_agent(policy, gamma, alpha)

    agent.train_mc(env, num_episodes)

    agent.plot_stats(IMG_PATH)

    agent.test_agent(env, 1111*2, 1, run_anime=False, store_gif=False)

    agent.save_state()