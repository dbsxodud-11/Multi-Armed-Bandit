import numpy as np
import math
import random
from scipy.stats import bernoulli
from tqdm import tqdm


class EpsGreedy :

    def __init__(self, eps) :
        self.name = "epsillon-greedy"
        self.eps = eps

    def execute(self, bandit, T) :

        reward_list = []
        for _ in tqdm(range(100)) :
            reward = self.run(bandit, T)
            reward_list.append(reward)
        reward_list = np.array(reward_list)

        return reward_list

    def run(self, bandit, T) :
        self.Q = [0.0 for _ in range(len(bandit))]
        self.N = [1 for _ in range(len(bandit))]

        rewards = [-1]

        for t in range(T) :
            if random.random() < self.eps :
                action = random.randint(0, len(bandit)-1)
            else :
                action = np.argmax(self.Q)
            reward = bernoulli.rvs(bandit[action], size=1)[0]
            rewards.append(reward)

            self.Q[action] += (rewards[-1] - self.Q[action]) / self.N[action]
            self.N[action] += 1

        return rewards