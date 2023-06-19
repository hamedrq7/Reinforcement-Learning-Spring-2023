import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
from time import sleep
from matplotlib import animation
from typing import List, Dict, Tuple
from tqdm import trange
import os 
import pickle 


IMG_PATH = 'imgs'

def mkdir(dir: str):
  if not os.path.exists(dir):
    os.makedirs(dir)

class eps_greedy_policy():
    def __init__(self, env, epsilon=0.1 ,is_mc: bool = False) -> None:
        self.env = env
        if is_mc:
            self.q_table = np.ones([env.observation_space.n, env.action_space.n]) / 10
            # self.q_table = np.random.uniform(size=[env.observation_space.n, env.action_space.n]) 
            
        else:
            self.q_table = np.zeros([env.observation_space.n, env.action_space.n]) 
        
        self.epsilon = epsilon

    def select_action(self, state): 
        # return action based epsilon greedy method using q_table and current state
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample() # (exploration)
        else:      
            # action = np.argmax(self.q_table[state]) # (exploitation)
            action = np.random.choice(np.flatnonzero(self.q_table[state] == self.q_table[state].max()))

        return action
    
class RL_agent():
    def __init__(self, policy: eps_greedy_policy, gamma=0.9, alpha=0.1) -> None:
        self.gamma = gamma
        self.policy = policy
        self.alpha = alpha

        # utility vars: 
        # plotting
        self.num_epcohs_history = []
        self.reward_history = []
        self.failed_dropoffs_history = []
        self.TD_error_history = []
        self.algo: str


    def generate_episode(self, env):
        state_action_pairs = []
        rewards = []
        
        done = False

        # reset env, first state
        state, info = env.reset()
        
        action_freq = np.zeros(10)
        
        while not done:
            # select action from policy
            action = self.policy.select_action(state)
            action_freq[action]+= 1

            # interact w/env
            next_state, reward, done, _, info = env.step(action, info)
            
            state_action_pairs.append([state, action])
            rewards.append(reward)

            state = next_state

            if len(rewards) % 10000 == 0: 
                print(len(rewards), action_freq)

        # return np.array(state_action_pairs), np.array(rewards)
        return state_action_pairs, rewards
        
    def train_mc(self, env: gym.Env, num_episodes=4000): 
        self.algo = 'MC'
        self.num_episodes = num_episodes

        self.N_visits = np.zeros((env.observation_space.n, env.action_space.n))

        for episode in trange(num_episodes): 
            # utility vars: 
            epoch = 0 
            failed_dropoffs = 0
            cumulative_reward = 0
            # cumulative_TD_error = 0
            
            # generate episode
            state_action_pairs, rewards = self.generate_episode(env)
            G = 0
            T = len(state_action_pairs)

            for t in range(T-1, -1, -1):
                state = state_action_pairs[t][0]
                action = state_action_pairs[t][1]

                G = self.gamma * G + rewards[t]

                if not state_action_pairs[t] in state_action_pairs[: t-1]:
                    # first visit of state-action pair 
                    self.N_visits[state][action] += 1
                    self.policy.q_table[state][action] += (1/self.N_visits[state][action]) \
                        * (G - self.policy.q_table[state][action])
                     
                epoch += 1

                # utilities: 
                cumulative_reward += rewards[t]
                # cumulative_TD_error += ((reward+self.gamma*bootstrap_q) - old_q_value)

                if rewards[t] == -10:
                    failed_dropoffs += 1
                
            self.num_epcohs_history.append(epoch)
            self.reward_history.append(cumulative_reward)
            self.failed_dropoffs_history.append(failed_dropoffs)
            # self.TD_error_history.append(cumulative_TD_error)

    def train_sarsa(self, env, num_episodes=4000): 
        self.algo = 'SARSA'
        self.num_episodes = num_episodes

        for episode in trange(num_episodes): 
            # utility vars: 
            epoch = 0 
            failed_dropoffs = 0
            cumulative_reward = 0
            cumulative_TD_error = 0

            done = False
            
            # reset env
            curr_state, info = env.reset()
            # get action (eps greedy based on q_table)
            action = self.policy.select_action(curr_state)
                
            while not done:
                
                # intract with env
                next_state, reward, done, _, info = env.step(action, info)

                # select action based on next_state 
                bootstrap_action = self.policy.select_action(next_state)

                old_q_value = self.policy.q_table[curr_state, action]
                bootstrap_q = self.policy.q_table[next_state, bootstrap_action]

                new_q_value = (1-self.alpha)*old_q_value+ \
                                    self.alpha*(reward+self.gamma*bootstrap_q)
                
                # update policy
                self.policy.q_table[curr_state, action] = new_q_value

                curr_state = next_state
                action = bootstrap_action 
                epoch += 1

                # utilities: 
                cumulative_reward += reward
                cumulative_TD_error += ((reward+self.gamma*bootstrap_q) - old_q_value)

                if reward == -10:
                    failed_dropoffs += 1
                
            self.num_epcohs_history.append(epoch)
            self.reward_history.append(cumulative_reward)
            self.failed_dropoffs_history.append(failed_dropoffs)
            self.TD_error_history.append(cumulative_TD_error)


    def train_q(self, env, num_episodes=4000):
        self.algo = 'Q' 
        self.num_episodes = num_episodes

        for episode in trange(num_episodes): 
            # utility vars: 
            epoch = 0 
            failed_dropoffs = 0
            cumulative_reward = 0
            cumulative_TD_error = 0

            done = False
            
            # reset env
            curr_state, info = env.reset()

            while not done:
                # get action (eps greedy based on q_table)
                action = self.policy.select_action(curr_state)
                
                # intract with env
                next_state, reward, done, _, info = env.step(action, info)

                old_q_value = self.policy.q_table[curr_state, action]
                max_q_value_for_next_state = np.max(self.policy.q_table[next_state])

                new_q_value = (1-self.alpha)*old_q_value+\
                    self.alpha*(reward+self.gamma*max_q_value_for_next_state)
                
                # update policy
                self.policy.q_table[curr_state, action] = new_q_value

                curr_state = next_state
                epoch += 1

                # utilities: 
                cumulative_reward += reward
                cumulative_TD_error += ((reward+self.gamma*max_q_value_for_next_state) - old_q_value)

                if reward == -10:
                    failed_dropoffs += 1
                
            self.num_epcohs_history.append(epoch)
            self.reward_history.append(cumulative_reward)
            self.failed_dropoffs_history.append(failed_dropoffs)
            self.TD_error_history.append(cumulative_TD_error)

    def test_agent(self, env, rand_seed: int,
                   num_episodes=1, run_anime=True,
                   store_gif=True): 
        
        num_epochs = 0

        # pirnt('testing ')
        for episode in trange(num_episodes): 
            # utils
            epoch = 1
            failed_dropoffs = 0
            cumulative_reward = 0
            done = False

            experience_buffer = []

            state, info = env.reset(seed=rand_seed)
            
            while not done: 
                # select action 
                action = self.policy.select_action(state)

                curr_time_stamp = {}
                curr_time_stamp['episode'] = episode
                curr_time_stamp['state'] = state
                curr_time_stamp['action'] = action
                curr_time_stamp['frame'] = env.render()
                curr_time_stamp['epoch'] = epoch
                
                # interact w/env:
                next_state, reward, done, _, info = env.step(action, info)

                curr_time_stamp['reward'] = reward
                cumulative_reward += reward

                if reward == -10:
                    failed_dropoffs += 1
                epoch += 1
                state = next_state

                experience_buffer.append(curr_time_stamp)

            num_epochs += epoch

            if store_gif and env.render_mode == 'rgb_array': 
                store_episode_as_gif(experience_buffer, IMG_PATH, filename=f'test_{self.algo}.gif')  

            if run_anime:
                run_animation(experience_buffer, env.render_mode=='rgb_array')

        # Print final results
        print("\n") 
        print(f"Mean # epochs per episode: {num_epochs / num_episodes}")

    def plot_stats(self, img_path):
        mkdir(img_path)
        self.exp_name = f'{self.algo}-gamma_{self.gamma}-{self.num_episodes}-epsilon_{self.policy.epsilon}'

        if len(self.reward_history) > 0:
            # Plot reward convergence
            plt.title(f"Cumulative reward per episode {self.exp_name}")
            plt.xlabel("Episode")
            plt.ylabel("Cumulative reward")
            plt.plot(self.reward_history)
            # plt.show()
            plt.savefig(f'{img_path}/Cumulative reward {self.exp_name}.png')
            plt.clf()
        
        if len(self.num_epcohs_history) > 0: 
            # Plot epoch convergence
            plt.title(f"# epochs per episode {self.exp_name}")
            plt.xlabel("Episode")
            plt.ylabel("# epochs")
            plt.plot(self.num_epcohs_history)
            # plt.show()
            plt.savefig(f'{img_path}/epochs per episode {self.exp_name}.png')
            plt.clf()

        if len(self.TD_error_history) > 0: 
            # Plot td error convergence
            plt.title(f"Cumulative TD error per episode {self.exp_name}")
            plt.xlabel("Episode")
            plt.ylabel("Cumulative TD error")
            plt.plot(self.TD_error_history)
            plt.savefig(f'{img_path}/Cumulative TD error {self.exp_name}.png')
            # plt.show()
            plt.clf()

    def save_state(self, path: str='stats'): 
        mkdir(path)

        self.exp_name = f'{self.algo}-gamma_{self.gamma}-{self.num_episodes}-epsilon_{self.policy.epsilon}'

        results = {
            'name': self.exp_name,
            'algo': self.algo,
            'num_episodes': self.num_episodes,
            'gamma': self.gamma,
            'policy': self.policy,
            'epsilon': self.policy.epsilon,
            'lr': self.alpha, 

            'reward_history': self.reward_history,
            'num_epcohs_history': self.num_epcohs_history,
            'TD_error_history': self.TD_error_history,
        }

        with open(f'{path}/{self.exp_name}.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # for loading pickle: 
        """
        with open('filename.pickle', 'rb') as handle:
            b = pickle.load(handle)
        """



class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.env.action_space = gym.spaces.Discrete(10)
        # ? 

    def step(self, action, info, debug=False):
        old_info = info
        if debug:
            print(old_info)
        if action == 6: # left-down
            if old_info['action_mask'][3] and old_info['action_mask'][0]: 
                _, _, _, _, _ = self.env.step(3) # go left 
                next_state, reward, terminated, truncated , info = self.env.step(0) # go down
            
            elif old_info['action_mask'][3] and not old_info['action_mask'][0]:
                next_state, reward, terminated, truncated , info = self.env.step(0) # go down
            
            elif not old_info['action_mask'][3] and old_info['action_mask'][0]:
                next_state, reward, terminated, truncated , info = self.env.step(3) # go left
                
            else:
                next_state, reward, terminated, truncated, info = self.env.step(3) # go left
                next_state, reward, terminated, truncated, info = self.env.step(0) # go down

        elif action == 7: # right-down
            if old_info['action_mask'][2] and old_info['action_mask'][0]: 
                _, _, _, _, _ = self.env.step(2) # go right 
                next_state, reward, terminated, truncated , info = self.env.step(0) # go down
            
            elif old_info['action_mask'][2] and not old_info['action_mask'][0]:
                next_state, reward, terminated, truncated , info = self.env.step(0) # go down
            
            elif not old_info['action_mask'][2] and old_info['action_mask'][0]:
                next_state, reward, terminated, truncated , info = self.env.step(2) # go right
                
            else:
                next_state, reward, terminated, truncated, info = self.env.step(2) # go right
                next_state, reward, terminated, truncated, info = self.env.step(0) # go down
        
        elif action == 8: # left-up
            if old_info['action_mask'][3] and old_info['action_mask'][1]: 
                _, _, _, _, _ = self.env.step(3) # go left 
                next_state, reward, terminated, truncated , info = self.env.step(1) # go up
            
            elif old_info['action_mask'][3] and not old_info['action_mask'][1]:
                next_state, reward, terminated, truncated , info = self.env.step(1) # go up
            
            elif not old_info['action_mask'][3] and old_info['action_mask'][1]:
                next_state, reward, terminated, truncated , info = self.env.step(3) # go left
                
            else:
                next_state, reward, terminated, truncated, info = self.env.step(3) # go left
                next_state, reward, terminated, truncated, info = self.env.step(1) # go up
                
        
        elif action == 9: # right-up
            if old_info['action_mask'][2] and old_info['action_mask'][1]: 
                _, _, _, _, _ = self.env.step(2) # go right 
                next_state, reward, terminated, truncated , info = self.env.step(1) # go up
            
            elif old_info['action_mask'][2] and not old_info['action_mask'][1]:
                next_state, reward, terminated, truncated , info = self.env.step(1) # go up
            
            elif not old_info['action_mask'][2] and old_info['action_mask'][1]:
                next_state, reward, terminated, truncated , info = self.env.step(2) # go right
                
            else:
                next_state, reward, terminated, truncated, info = self.env.step(2) # go rught
                next_state, reward, terminated, truncated, info = self.env.step(1) # go up
        
        else: # normal actions (0-5)

            next_state, reward, terminated, truncated , info = self.env.step(action)
        
        return next_state, reward, terminated, truncated , info


class BasicWrapper22(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.env.action_space = gym.spaces.Discrete(10)
        # ? 

    def step(self, action):
        if action == 6: # left-down
            _, _, _, _, _ = self.env.step(3) # go left 
            next_state, reward, terminated, truncated , info = self.env.step(0) # go down
        

        elif action == 7: # right-down
            _, _, _, _, _ = self.env.step(2) # go right 
            next_state, reward, terminated, truncated , info = self.env.step(0) # go down
            
        elif action == 8: # left-up
            _, _, _, _, _ = self.env.step(3) # go left 
            next_state, reward, terminated, truncated , info = self.env.step(1) # go up
            
        
        elif action == 9: # right-up
            _, _, _, _, _ = self.env.step(2) # go right 
            next_state, reward, terminated, truncated , info = self.env.step(1) # go up

        else: # normal actions (0-5)

            next_state, reward, terminated, truncated , info = self.env.step(action)
        
        return next_state, reward, terminated, truncated , info




def run_animation(experience_buffer, ispic: bool):
    """Function to run animation"""
    time_lag = 0.05  # Delay (in s) between frames
    for experience in experience_buffer:
        # Plot frame
        if ispic:
            clear_output(wait=True)
            plt.imshow(experience['frame'])
            plt.axis('off')
            plt.show()
        else:
            print(experience['frame'])

        # Print console output
        print(f"Episode: {experience['episode']}/{experience_buffer[-1]['episode']}")
        print(f"Epoch: {experience['epoch']}/{experience_buffer[-1]['epoch']}")
        print(f"State: {experience['state']}")
        print(f"Action: {experience['action']}")
        print(f"Reward: {experience['reward']}")
        # Pauze animation
        sleep(time_lag)

def store_episode_as_gif(experience_buffer, path='./', filename='animation.gif'):
    """Store episode as gif animation"""
    fps = 5   # Set framew per seconds
    dpi = 300  # Set dots per inch
    interval = 50  # Interval between frames (in ms)

    # Retrieve frames from experience buffer
    frames = []
    for experience in experience_buffer:
        frames.append(experience['frame'])

    # Fix frame size
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=dpi)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    # Generate animation
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=interval)

    # Save output as gif
    anim.save(path + filename, writer='imagemagick', fps=fps)



def plot_frame(env_rendered):
    plt.imshow(env_rendered)
    plt.axis("off")
    plt.show()



# ## # # # SIM 
