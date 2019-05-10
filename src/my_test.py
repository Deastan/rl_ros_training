#!/usr/bin/env python

import numpy as np
import gym
import matplotlib.pyplot as plt

# import gym
import time
# import numpy
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

from gym import envs


if __name__ == '__main__':

    rospy.init_node('My_test', anonymous=True, log_level=rospy.WARN)

#Part********************************* 
    # env = gym.make('CartPole-v0')
    # env.reset()
    # for _ in range(1000):
    #     env.render()
    #     env.step(env.action_space.sample()) # take a random action
    # env.close()
    # import gym

#Part*********************************

    # env_name = "CartPole-v0"
    # env = gym.make(env_name)
    # print('Observation space', env.observation_space)
    # print('Action space', env.action_space)
    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(20):
    #         env.render()
    #         # print(observation)
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         # print(reward)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t+1))
    #             break
    # env.close()

#Part*********************************


    # Import and initialize Mountain Car Environment
    env = gym.make('MountainCar-v0')
    env.reset()

    # Define Q-learning function
    def QLearning(env, learning, discount, epsilon, min_eps, episodes):
        # Determine size of discretized state space
        num_states = (env.observation_space.high - env.observation_space.low)*\
                        np.array([10, 100])
        num_states = np.round(num_states, 0).astype(int) + 1
        
        # Initialize Q table
        Q = np.random.uniform(low = -1, high = 1, 
                            size = (num_states[0], num_states[1], 
                                    env.action_space.n))
        
        # Initialize variables to track rewards
        reward_list = []
        ave_reward_list = []
        
        # Calculate episodic reduction in epsilon
        reduction = (epsilon - min_eps)/episodes
        
        # Run Q learning algorithm
        for i in range(episodes):
            # Initialize parameters
            done = False
            tot_reward, reward = 0,0
            state = env.reset()
            
            # Discretize state
            state_adj = (state - env.observation_space.low)*np.array([10, 100])
            state_adj = np.round(state_adj, 0).astype(int)
        
            while done != True:   
                # Render environment for last five episodes
                if i >= (1):#episodes - 20):
                    env.render()
                    
                # Determine next action - epsilon greedy strategy
                if np.random.random() < 1 - epsilon:
                    action = np.argmax(Q[state_adj[0], state_adj[1]]) 
                else:
                    action = np.random.randint(0, env.action_space.n)
                    
                # Get next state and reward
                state2, reward, done, info = env.step(action) 
                
                # Discretize state2
                state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
                state2_adj = np.round(state2_adj, 0).astype(int)
                
                #Allow for terminal states
                if done and state2[0] >= 0.5:
                    Q[state_adj[0], state_adj[1], action] = reward
                    
                # Adjust Q value for current state
                else:
                    delta = learning*(reward + 
                                    discount*np.max(Q[state2_adj[0], 
                                                    state2_adj[1]]) - 
                                    Q[state_adj[0], state_adj[1],action])
                    Q[state_adj[0], state_adj[1],action] += delta
                                        
                # Update variables
                tot_reward += reward
                state_adj = state2_adj
            
            # Decay epsilon
            if epsilon > min_eps:
                epsilon -= reduction
            
            # Track rewards
            reward_list.append(tot_reward)
            
            if (i+1) % 100 == 0:
                ave_reward = np.mean(reward_list)
                ave_reward_list.append(ave_reward)
                reward_list = []
                
            if (i+1) % 100 == 0:    
                print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
                
        env.close()
        
        return ave_reward_list

    # Run Q-learning algorithm
    rewards = QLearning(env, 0.2, 0.9, 0.8, 0, 5000)

    # Plot Rewards
    plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('rewards.jpg')     
    plt.show()
    # plt.close() 