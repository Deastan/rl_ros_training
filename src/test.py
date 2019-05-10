#!/usr/bin/env python

import gym
import numpy
import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from gym import envs



if __name__ == '__main__':

    rospy.init_node('My_test',
                    anonymous=True, log_level=rospy.WARN)
    
    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
    env.close()

    