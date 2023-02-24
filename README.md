# [Efficient Bayesian Ultra-Q Learning for Multi-Agent Games](Bayesian_Ultra_Q.pdf)

Ward Gauderis - Fabian Denoodt - Bram Silue - Pierre Vanvolsem

Vrije Universiteit Brussel

ALA2023: Adaptive and Learning Agents Workshop 2023

---

## Abstract

This paper presents Bayesian Ultra-Q Learning, a variant of Q-Learning adapted for solving multi-agent games with independent learning agents. Bayesian Ultra-Q Learning is an extension of the Bayesian Hyper-Q Learning algorithm proposed by Tesauro that is more efficient for solving adaptive multi-agent games. While Hyper-Q agents merely update the Q-table corresponding to a single state, Ultra-Q leverages the information that similar states most likely result in similar rewards, and therefore updates the Q-values of nearby states as well. 

We assess the performance of our Bayesian Ultra-Q Learning algorithm against three variants of Hyper-Q as defined by Tesauro, and against Infinitesimal Gradient Ascent (IGA) and Policy Hill Climbing (PHC) agents. We do so by evaluating the agents in the zero-sum game of rock-paper-scissors and a cooperative stochastic hill-climbing game. In rock-paper-scissors, games of Bayesian Ultra-Q agents against IGA agents end in draws where, averaged over time, all players play the Nash Equilibrium, meaning no player can exploit another. Against PHC, neither Bayesian Ultra-Q nor Hyper-Q agents are able to win on average, which goes against the findings of Tesauro.

In the cooperation game, Bayesian Ultra-Q converges in the direction of an optimal joint strategy and vastly outperforms all other algorithms including Hyper-Q, which are unsuccessful in finding a strong equilibrium. 