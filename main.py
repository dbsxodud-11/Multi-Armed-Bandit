import argparse
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

from algorithms.EpsGreedy import *
from algorithms.EpsGreedy_Improved import *
from algorithms.UCB import *
from algorithms.Thompson_Sampling import *

def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--number-of-arms", type=int, default=5, help="number of arms")
    parser.add_argument("--trials", type=int, default=10000, help="trials")
    parser.add_argument("--algo", type=str, default="epsilon_greedy_improved", help="choose algorithm to train")
    parser.add_argument("--output_path", type=str, default="./_plots/epsilon-greedy-improved/", help="output path to store results")
    parser.add_argument("--seed", type=int, default=10, help="seed for random module")
    args = parser.parse_args()
    return args

def show_bandit_information(args) :
    print("Bandit Information...")
    print(f"Number of Arms : {args.number_of_arms}")
    print(f"Trials : {args.trials}")
    print()
    random.seed(args.seed)
    bandit = [random.random() for _ in range(args.number_of_arms)]
    # bandit = [0.4, 0.6, 0.8]
    print(bandit)
    return bandit

def show_algorithm_information(args) :
    print("Algorithm Information...")
    print(f"Algorithm Name : {args.algo}")
    print()

    if args.algo == "epsillon_greedy" :
        eps = 0.3
        algo = EpsGreedy(eps)
    elif args.algo == "epsilon_greedy_improved" :
        algo = EpsGreedyImproved()
    elif args.algo == "UCB" :
        c = 5
        algo = UCB(c)
    elif args.algo == "UCB-Improved" :
        algo = UCBImproved()
    else :
        algo = Thompson_Sampling()
    return algo

def visualize(path, reward, T) :
    style.use('ggplot')

    x = []
    for i in list(range(T)):
        x.append(math.log10(i+1))

    # 1. Reward
    fig1 = plt.figure()
    plt.title("Epsilon_Greedy_Improved")
    plt.xlabel("trials")
    plt.plot(x, np.mean(reward[:,1:], axis=0), color="mediumseagreen", linewidth=2.0, label="reward")
    plt.legend()
    plt.savefig(path + "reward.png") 

    # 2. Cumulative Reward
    cum_reward = np.cumsum(np.mean(reward[:,1:], axis=0))
    fig2 = plt.figure()
    plt.title("Epsilon_Greedy_Improved")
    plt.xlabel("trials")
    plt.plot(x, cum_reward, linewidth=2.0, color="darkgray", label="cumulative_reward")
    plt.legend()
    plt.savefig(path + "cumulative_reward.png")

    df = pd.DataFrame(cum_reward)
    df.to_csv("./_data/epsilon_greedy_improved.csv")

def visualize_all(T) :
    style.use('ggplot')

    x = []
    for i in list(range(T)):
        x.append(math.log10(i+1))

    eps_greedy_improved = pd.read_csv("./_data/epsilon_greedy_improved.csv")
    ucb_improved = pd.read_csv("./_data/ucb_improved.csv")
    thompson_sampling = pd.read_csv("./_data/thompson_sampling.csv")

    plt.title("Performance")
    plt.xlabel("trials")
    plt.plot(x, eps_greedy_improved.values[:,1], linewidth=2.0, color="indianred", label="Epsilon-Greedy")
    plt.plot(x, ucb_improved.values[:,1], linewidth=2.0, color="mediumpurple", label="UCB")
    plt.plot(x, thompson_sampling.values[:,1], linewidth=2.0, color="mediumblue", label="Thompson Sampling")
    plt.legend()
    plt.savefig("./_plots/comparison.png")


def main(args) :
    bandit = show_bandit_information(args)
    algorithm = show_algorithm_information(args)

    reward = algorithm.execute(bandit, args.trials)

    visualize(args.output_path, reward, args.trials)


if __name__ == "__main__" :
    args = parse_args()
    # main(args)
    visualize_all(args.trials)