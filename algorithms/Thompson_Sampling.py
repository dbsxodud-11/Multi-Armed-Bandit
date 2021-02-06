import numpy as np
import math
import random
from scipy.stats import bernoulli, beta
from tqdm import tqdm


class Thompson_Sampling :

    def __init__(self) :
        self.name = "Thompson_Sampling"

    def execute(self, bandit, T) :

        reward_list = []
        for _ in tqdm(range(100)) :
            reward = self.run(bandit, T)
            reward_list.append(reward)
        reward_list = np.array(reward_list)

        return reward_list

    def run(self, bandit, T) :
        self.alpha = np.array([1.0 for _ in range(len(bandit))])
        self.beta = np.array([1.0 for _ in range(len(bandit))])

        rewards = [-1]

        for t in range(T) :
            q_values = []
            for i in range(len(bandit)) :
                q_value = beta.rvs(self.alpha[i], self.beta[i], size=1)
                q_values.append(q_value)
            
            action = np.argmax(q_values)
            reward = bernoulli.rvs(bandit[action], size=1)[0]
            rewards.append(reward)

            self.alpha[action] += reward
            self.beta[action] += 1 - reward
        return rewards