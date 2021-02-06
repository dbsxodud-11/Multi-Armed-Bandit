import numpy as np
import math
import random
from scipy.stats import bernoulli
from tqdm import tqdm


class UCB :

    def __init__(self, c) :
        self.name = "UCB"
        self.c = c

    def execute(self, bandit, T) :

        reward_list = []
        for _ in tqdm(range(100)) :
            reward = self.run(bandit, T)
            reward_list.append(reward)
        reward_list = np.array(reward_list)

        return reward_list

    def run(self, bandit, T) :
        self.Q = np.array([0.0 for _ in range(len(bandit))])
        self.N = np.array([0 for _ in range(len(bandit))])

        rewards = [-1]

        for t in range(T) :
            if t < len(bandit) :
                reward = bernoulli.rvs(bandit[t], size=1)[0]
                rewards.append(reward)

                self.Q[t] = reward
                self.N[t] = 1

            else :
                action = np.argmax(self.Q + self.c * np.sqrt(np.repeat(math.log(t), len(bandit)) / self.N))
                reward = bernoulli.rvs(bandit[action], size=1)[0]
                rewards.append(reward)

                self.Q[action] += (rewards[-1] - self.Q[action]) / self.N[action]
                self.N[action] += 1

        return rewards

class UCBImproved :

    def __init__(self) :
        super(UCBImproved, self).__init__()
        self.name = "UCB-Improved"

    def execute(self, bandit, T) :

        reward_list = []
        for _ in tqdm(range(100)) :
            reward = self.run(bandit, T)
            reward_list.append(reward)
        reward_list = np.array(reward_list)

        return reward_list

    def run(self, bandit, T) :
        self.Q = np.array([0.0 for _ in range(len(bandit))])
        self.N = np.array([0 for _ in range(len(bandit))])

        rewards = [-1]

        for t in range(T) :
            if t < len(bandit) :
                reward = bernoulli.rvs(bandit[t], size=1)[0]
                rewards.append(reward)
                
                self.Q[t] = reward
                self.N[t] = 1

            else :
                action = np.argmax(self.Q + math.log(t) * np.sqrt(np.repeat(math.log(t), len(bandit)) / self.N))
                reward = bernoulli.rvs(bandit[action], size=1)[0]
                rewards.append(reward)

                self.Q[action] += (rewards[-1] - self.Q[action]) / self.N[action]
                self.N[action] += 1

        return rewards 