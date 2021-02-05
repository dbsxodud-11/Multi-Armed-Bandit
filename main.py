import argparse
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib import style

from algorithms.EpsGreedy import *

def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--number-of-arms", type=int, default=5, help="number of arms")
    parser.add_argument("--trials", type=int, default=10000, help="trials")
    parser.add_argument("--algo", type=str, default="epsillon_greedy", help="choose algorithm to train")
    parser.add_argument("--output_path", type=str, default="./_plots/epsillon-greedy/", help="output path to store results")
    args = parser.parse_args()
    return args

def show_bandit_information(args) :
    print("Bandit Information...")
    print(f"Number of Arms : {args.number_of_arms}")
    print(f"Trials : {args.trials}")
    print()

    bandit = [random.random() for _ in range(args.number_of_arms)]
    # bandit = [0.4, 0.6, 0.8]
    print(bandit)
    return bandit

def show_algorithm_information(args) :
    print("Algorithm Information...")
    print(f"Algorithm Name : {args.algo}")
    print()

    if args.algo == "epsillon_greedy" :
        eps = 0.1
        algo = EpsGreedy(eps)
    return algo

def visualize(path, reward, T) :
    style.use('ggplot')

    x = []
    for i in list(range(T)):
        x.append(math.log10(i+1))

    # 1. Reward
    fig1 = plt.figure()
    plt.title("Epsilon-Greedy")
    plt.xlabel("trials")
    plt.plot(x, np.mean(reward[:,1:], axis=0), color="mediumseagreen", linewidth=2.0, label="reward")
    plt.legend()
    plt.savefig(path + "reward.png") 

    # fig2 = plt.figure()
    # plt.title("Epsilon-Greedy")
    # plt.xlabel("trials")
    # for i in range(len(action)) :
    #     plt.plot(np.where(action==i, 1, 0), linewidth=2.0, label=f"Arm{i+1}")
    # plt.legend()
    # plt.savefig(path + "action.png")

def main(args) :
    bandit = show_bandit_information(args)
    algorithm = show_algorithm_information(args)

    reward = algorithm.execute(bandit, args.trials)

    visualize(args.output_path, reward, args.trials)


if __name__ == "__main__" :
    args = parse_args()
    main(args)