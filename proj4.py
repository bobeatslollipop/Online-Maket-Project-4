import math
import csv
import os
import random
import numpy as np
import matplotlib.pyplot as plt


random.seed(2)

def unif_val(k, actions):
    val = random.random()
    return actions[int(val * k)]

def squared_cdf(k, actions):
    val = math.sqrt(random.random())
    return actions[int(val * k)]


# (k, V) -> (choice, prob[k])
def EW(k, epsilon, actions, V):
    maxi = max(V[-1])
    Vtemp = [j - maxi for j in V[-1]]
    pitemp = [(1+epsilon) ** Vtemp[j]
              for j in range(len(actions))]
    denom = sum(pitemp)
    pitemp = [pitemp[j] / denom for j in range(len(actions))]
    do = random.choices([j for j in range(k)], weights=pitemp, k=1)
    return do[0], pitemp

# (...) -> (do, prob)
# Assuming n_buyers > n_items
def simulate_auction(k, actions, n_buyers, n_items, V, payoff, epsilon, bids):
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
    payoff.append(payoff[-1] + v[do])

    return do, prob

def exploit_auction(k, actions, n_buyers, n_items, V, payoff, epsilon, means, bids):
    # Calculate means
    lowbid = min(bids)
    highbid = max(bids)
    means[0] = (means[0] * (len(V)-1) + lowbid) / len(V)
    means[1] = (means[1] * (len(V)-1) + highbid) / len(V)

    # Randomly set reserve
    do, prob = EW(k, epsilon, actions, V)
    do = int(np.mean(means) * k)

    # Simulate different reserve prices
    # (n+1) price auction if there are n buyers
    v = [0 for j in range(k)]
    for j in range(k):
        reserve = actions[j]
        if reserve < bids[n_items]: # original price >= reserve
            v[j] = bids[n_items] * n_items
        else: # original price < reserve
            cutoff = 0
            for i in range(n_buyers):
                if bids[i] < reserve:
                    cutoff = i
                    break
            v[j] = reserve * cutoff
    V.append([V[-1][j] + v[j] for j in range(k)])
    payoff.append(V[-1][do])

    return do, prob
    

def simulate_introduction(k, actions, possible_bids, n_buyers, n_items, V, payoff, epsilon, bids):
    # Randomly set reserve
    do, prob = EW(k, epsilon, actions, V)

    # Simulate different reserve prices
    # (n+1) price auction if there are n buyers
    v = [0 for j in range(k)]
    for j in range(k):
        reserve = actions[j]
        if reserve <= sum(bids): # original price >= reserve
            v[j] = max(reserve * 2 - sum(bids), 0)
        else: # original price < reserve
            v[j] = 0
    V.append([V[-1][j] + v[j] for j in range(k)])
    payoff.append(payoff[-1] + v[do])

    return do, prob

# 1 item, uniform value, 2 players
def test_convergence(N, n, k, n_buyers, epsilon_list):
    actions = [j / (k-1) for j in range(k)]
    probss = []
    bidss = []
    for t in range(N):
        bids = []
        for round in range(n):
            bid = [unif_val(k, actions) for b in range(n_buyers)]
            bid.sort(reverse=True)
            bids.append(bid)
        bidss.append(bids)
    for epsilon in epsilon_list:
        regrets = []
        payoffs = []
        probs = []
        for t in range(N):
            V = [[0 for j in range(k)]]
            regret = [0]
            payoff = [0]

            for round in range(n):
                do, prob = simulate_auction(k, actions, n_buyers, 1, V, payoff, epsilon, bidss[t][round])
                regret.append((V[-1][25] - payoff[-1]) / (round+1))

            regrets.append(regret)
            payoffs.append(payoff)
            probs.append(prob)
        probs_avg = np.mean(probs, axis=0)
        probss.append(probs_avg[25])
    probss = np.array(probss)

    def plot():
        # prob vs epsilon
        plt.figure()
        plt.plot(epsilon_list, probss)
        plt.xlabel("epsilon")
        plt.ylabel("probability of selecting optimal reserve")
        plt.title("probability of optimal reserve vs epsilon")
        plt.savefig('conv/prob, n_buyers={}, n={}'.format(n_buyers, n))

    plot()
    return

def test_exploit(N, n, k, n_buyers, epsilon):
    actions = [j / (k-1) for j in range(k)]
    probs1 = []
    probs2 = []
    regrets1 = []
    regrets2 = []
    doss1 = []
    doss2 = [] 

    for t in range(N):
        bids = []
        for round in range(n):
            bid = [unif_val(k, actions) for b in range(n_buyers)]
            bid.sort(reverse=True)
            bids.append(bid)
        V1 = [[0 for j in range(k)]]
        V2 = [[0 for j in range(k)]]
        means = [0, 0]
        payoff1 = [0]
        payoff2 = [0]
        regret1 = []
        regret2 = []
        dos1 = []
        dos2 = []

        for round in range(n):
            do1, prob1 = simulate_auction(k, actions, n_buyers, 1, V1, payoff1, epsilon, bids[round])
            do2, prob2 = exploit_auction(k, actions, n_buyers, 1, V2, payoff2, epsilon, means, bids[round])
            regret1.append((V1[-1][25] - payoff1[-1]) / (round+1))
            regret2.append((V2[-1][25] - payoff2[-1]) / (round+1))
            dos1.append(do1)
            dos2.append(do2)

        probs1.append(prob1)
        probs2.append(prob2)
        regrets1.append(regret1)
        regrets2.append(regret2)
        doss1.append(dos1)
        doss2.append(dos2)
        
    probs_avg1 = np.mean(probs1, axis=0)
    probs_avg2 = np.mean(probs2, axis=0)
    regrets_avg1 = np.mean(regrets1, axis=0)
    regrets_avg2 = np.mean(regrets2, axis=0)
    doss_avg1 = np.mean(doss1, axis=0)
    doss_avg2 = np.mean(doss2, axis=0)
    V1 = np.array(V1)
    

    def plot():
        # Regret
        plt.figure()
        plt.plot(regrets_avg1[1:], label='learning')
        plt.plot(regrets_avg2[1:], label='SE')
        plt.xticks(np.arange(1, n, step=n//10))
        plt.xlabel("round")
        plt.ylabel("per round regret")
        plt.title("regret vs round, {} buyers".format(n_buyers))
        plt.legend()
        plt.savefig('exploit/Reg, n={}, n_buyers={}'.format(n, n_buyers))

        # payoff vs round
        plt.figure()
        plt.plot([doss_avg1[j] / (k-1) for j in range(n)], label='learning')
        plt.plot([doss_avg2[j] / (k-1) for j in range(n)], label='SE')
        plt.plot([actions[25] for j in range(n)])
        plt.xticks(np.arange(1, n, step=n//10))
        plt.xlabel("round")
        plt.ylabel("reserve price taken")
        plt.title("reserve price vs round, {} buyers".format(n_buyers))
        plt.legend()
        plt.savefig('exploit/payoff2, n={}, n_buyers={}'.format(n, n_buyers))

    plot()
    return

# 1 item, uniform value
def test_buyers(N, n, k, n_buyers_total, epsilon):
    def test(n_buyers):
        regrets = []
        payoffs = []
        probs = []
        for t in range(N):
            actions = [j / (k-1) for j in range(k)]
            V = [[0 for j in range(k)]]
            regret = [0]
            payoff = [0]

            for round in range(n):
                bids = [unif_val(k, actions) for b in range(n_buyers)]
                bids.sort(reverse=True)
                do, prob = simulate_auction(k, actions, n_buyers, 1, V, payoff, epsilon, bids)
                regret.append((V[-1][25] - V[-1][do]) / (round+1))

            regrets.append(regret)
            payoffs.append(payoff)
            probs.append(prob)
        payoffs_avg = np.mean(payoffs, axis=0)
        regrets_avg = np.mean(regrets, axis=0)
        probs_avg = np.mean(probs, axis=0)
        return probs_avg, regrets_avg
    probss = [0, 0]
    regretss = [0, 0]
    for n_buyers in range(2, n_buyers_total):
        prob, reg = test(n_buyers)
        probss.append(prob)
        regretss.append(reg)

    # Probability
    plt.figure()
    for n_buyers in range(2, n_buyers_total):
        plt.plot(np.arange(0, 1, step=1/k), probss[n_buyers], label='{} buyers'.format(n_buyers))
    plt.xticks(np.arange(0, 1, step=10/k))
    plt.xlabel("choice")
    plt.ylabel("probability")
    plt.title("Selction probability")
    plt.legend()
    plt.savefig('buyers/prob, n={}'.format(n))

    # Regret
    plt.figure()
    for n_buyers in range(2, n_buyers_total):
        plt.plot(regretss[n_buyers][1:], label='{} buyers'.format(n_buyers))
    plt.xticks(np.arange(1, n, step=n//10))
    plt.xlabel("round")
    plt.ylabel("per round regret")
    plt.title("regret vs round, n={}".format(n))
    plt.legend()
    plt.savefig('buyers/Reg, n={}'.format(n))

def test_distribution(N, n, k, n_buyers, epsilon):
    regrets = []
    payoffs = []
    probs = []
    for t in range(N):
        actions = [j / (k-1) for j in range(k)]
        V = [[0 for j in range(k)]]
        regret = [0]
        payoff = [0]

        for round in range(n):
            bids = [squared_cdf(k, actions) for b in range(n_buyers)]
            bids.sort(reverse=True)
            do, prob = simulate_auction(k, actions, n_buyers, 1, V, payoff, epsilon, bids)
            regret.append((V[-1][25] - V[-1][do]) / (round+1))

        regrets.append(regret)
        payoffs.append(payoff)
        probs.append(prob)
    
    payoffs_avg = np.mean(payoffs, axis=0)
    regrets_avg = np.mean(regrets, axis=0)
    probs_avg = np.mean(probs, axis=0)

    plt.figure()
    plt.plot(np.arange(0, 1, step=1/k), probs_avg, label='F(x)=x^2')
    plt.xticks(np.arange(0, 1, step=10/k))
    plt.xlabel("choice")
    plt.ylabel("probability")
    plt.title("Selction probability")
    plt.savefig('distro/prob')

def test_introduction(N, n, k, epsilon):
    # regrets = []
    payoffs = []
    probs = []
    for t in range(N):
        actions = [2*j / (k-1) for j in range(k)]
        possible_bids = [j / (k-1) for j in range(k)]
        V = [[0 for j in range(k)]]
        regret = [0]
        payoff = [0]

        for round in range(n):
            bids = [unif_val(k, possible_bids) for b in range(2)]
            bids.sort(reverse=True)
            do, prob = simulate_introduction(k, actions, possible_bids, 2, 1, V, payoff, epsilon, bids)
            # regret.append((V[-1][25] - V[-1][do]) / (round+1))

        # regrets.append(regret)
        payoffs.append(payoff)
        probs.append(prob)
    payoffs_avg = np.mean(payoffs, axis=0)
    # regrets_avg = np.mean(regrets, axis=0)
    probs_avg = np.mean(probs, axis=0)

    plt.figure()
    plt.plot(np.arange(0, 2, step=2/k), probs_avg)
    plt.xticks(np.arange(0, 2, step=10/k))
    plt.xlabel("choice")
    plt.ylabel("probability")
    plt.title("Selction probability")
    plt.legend()
    plt.savefig('intro/prob, n={}'.format(n))

    # Regret
    plt.figure()
    plt.plot(payoffs_avg[1:])
    plt.xticks(np.arange(1, n, step=n//10))
    plt.xlabel("round")
    plt.ylabel("payoff")
    plt.title("payoffs vs round, n={}".format(n))
    plt.legend()
    plt.savefig('intro/Reg, n={}'.format(n))

def test_introduction2(N, n, k, epsilon):
    # regrets = []
    payoffs = []
    probs = []
    for t in range(N):
        actions = [2*j / (k-1) for j in range(k)]
        possible_bids = [j / (k-1) for j in range(k)]
        V = [[0 for j in range(k)]]
        regret = [0]
        payoff = [0]

        for round in range(n):
            bids = [unif_val(k, possible_bids), squared_cdf(k, possible_bids)]
            bids.sort(reverse=True)
            do, prob = simulate_introduction(k, actions, possible_bids, 2, 1, V, payoff, epsilon, bids)
            # regret.append((V[-1][25] - V[-1][do]) / (round+1))

        # regrets.append(regret)
        payoffs.append(payoff)
        probs.append(prob)
    payoffs_avg = np.mean(payoffs, axis=0)
    # regrets_avg = np.mean(regrets, axis=0)
    probs_avg = np.mean(probs, axis=0)

    plt.figure()
    plt.plot(np.arange(0, 2, step=2/k), probs_avg)
    plt.xticks(np.arange(0, 2, step=10/k))
    plt.xlabel("choice")
    plt.ylabel("probability")
    plt.title("Selction probability")
    plt.legend()
    plt.savefig('intro/prob2, n={}'.format(n))

    # Regret
    plt.figure()
    plt.plot(payoffs_avg[1:])
    plt.xticks(np.arange(1, n, step=n//10))
    plt.xlabel("round")
    plt.ylabel("payoff")
    plt.title("payoffs vs round, n={}".format(n))
    plt.legend()
    plt.savefig('intro/payoff2, n={}'.format(n))



# test_buyers(N=40, n=1000, k=51, n_buyers_total=5, epsilon=1)

# test_convergence(N=40, n=1000, k=51, n_buyers=2, epsilon_list=np.arange(1, 50, step=1))

# test_distribution(N=40, n=1000, k=51, n_buyers=2, epsilon=1)

# test_exploit(N=40, n=1000, k=51, n_buyers=2, epsilon=1)

test_introduction2(N=40, n=2000, k=51, epsilon=1)
