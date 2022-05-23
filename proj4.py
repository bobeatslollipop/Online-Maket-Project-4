import math
import csv
import os
import random
import numpy as np
import matplotlib.pyplot as plt


random.seed(1)

def unif_val(k, actions):
    val = random.random()
    return actions[int(val * k)]


# (k, V) -> (choice, prob[k])
def EW(k, epsilon, actions, V):
    pitemp = [(1+epsilon) ** (V[-1][j])
              for j in range(len(actions))]
    denom = sum(pitemp)
    pitemp = [pitemp[j] / denom for j in range(len(actions))]
    do = random.choices([j for j in range(k)], weights=pitemp, k=1)
    return do[0], pitemp

# (...) -> (do, prob)
# Assuming n_buyers > n_items
def simulate_auction(k, actions, n_buyers, n_items, val_func, V, payoff, epsilon):
    bids = [val_func(k, actions) for b in range(n_buyers)]
    bids.sort(reverse=True)

    # Randomly set reserve
    do, prob = EW(k, epsilon, actions, V)

    # Simulate different reserve prices
    # (n+1) price auction if there are n buyers
    v = [0 for j in range(k)]
    for j in range(k):
        reserve = actions[j]
        if reserve <= bids[n_items]: # original price >= reserve
            v[j] = bids[n_items] * n_items
        else: # original price < reserve
            cutoff = 0
            for i in range(n_buyers):
                if bids[i] < reserve:
                    cutoff = i
                    break
            v[j] = reserve * cutoff
    V.append([V[-1][j] + v[j] for j in range(k)])
    maxi = max(V[-1])
    for j in range(k):
        V[-1][j] -= maxi
    payoff.append(payoff[-1] + v[do])

    return do, prob

# 1 item, uniform value, 2 players
def test_convergence(N, k, n_buyers, epsilon_list, title="test_conv"):
    avg_conv_rounds = []
    for epsilon in epsilon_list:
        conv_rounds = []
        for t in range(N):
            actions = [j / k for j in range(k)]
            V = [[0 for j in range(k)]]
            payoff = [0]
            round = 0

            while True:
                round += 1
                do, prob = simulate_auction(k, actions, n_buyers, 1, unif_val, V, payoff, epsilon)
                if prob[k//2] >= 0.98:
                    conv_rounds.append(round)
                    break

        avg_conv_rounds.append(np.mean(conv_rounds))
    def plot():
        # convergence speed vs epsilon
        plt.figure()
        plt.plot(epsilon_list, avg_conv_rounds)
        plt.xlabel("epsilon")
        plt.ylabel("rounds until convergence")
        plt.title("convergence speed vs. epsilon, {} buyers".format(n_buyers))
        plt.savefig('conv/Prob, n={}, n_buyers={}'.format(n, n_buyers))


    plot()
    return

# 1 item, uniform value
def test_buyers(N, n, k, n_buyers, epsilon, title="test_buyers"):
    regrets = []
    payoffs = []
    probs = []
    for t in range(N):
        actions = [j / k for j in range(k)]
        V = [[0 for j in range(k)]]
        regret = [0]
        payoff = [0]

        for round in range(n):
            do, prob = simulate_auction(k, actions, n_buyers, 1, unif_val, V, payoff, epsilon)
            regret.append((V[-1][k//2] - V[-1][do]) / (round+1))

        regrets.append(regret)
        payoffs.append(payoff)
        probs.append(prob)
    payoffs_avg = np.mean(payoffs, axis=0)
    regrets_avg = np.mean(regrets, axis=0)
    probs_avg = np.mean(probs, axis=0)
    def plot():
        # Probability
        plt.figure()
        plt.plot(np.arange(0, 1, step=1/k), probs_avg)
        plt.xticks(np.arange(0, 1, step=10/k))
        plt.xlabel("choice")
        plt.ylabel("probability")
        plt.title("Selction probability, {}, {} buyers".format(title, n_buyers))
        plt.savefig('buyers/Prob, n={}, n_buyers={}'.format(n, n_buyers))

        # Regret
        plt.figure()
        plt.plot(regrets_avg[1:])
        plt.xticks(np.arange(1, n, step=n//10))
        plt.xlabel("round")
        plt.ylabel("per round regret")
        plt.title("regret vs round, {}, {} buyers".format(title, n_buyers))
        plt.savefig('buyers/Reg, n={}, n_buyers={}'.format(n, n_buyers))


    plot()

# test_buyers(N=40, n=1000, k=50, n_buyers=2, epsilon=1)
# test_buyers(N=40, n=1000, k=50, n_buyers=3, epsilon=1)
# test_buyers(N=40, n=1000, k=50, n_buyers=4, epsilon=1)

test_convergence(N=40, k=50, n_buyers=2, epsilon_list=np.arange(0.1,5,step=0.1))
