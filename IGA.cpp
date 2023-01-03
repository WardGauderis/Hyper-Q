#include "IGA.h"
#include "RockPaperScissors.h"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <memory>
#include <iostream>
#include <cmath>

// https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Nash+Convergence+of+Gradient+Dynamics+in+General-Sum+Games.&btnG=


IGA::IGA(float step_size_) {
    step_size = step_size_;

    srand(static_cast<unsigned int>(time(nullptr))); 
    policy[0] = static_cast<float>(rand()) / RAND_MAX;

    srand(static_cast<unsigned int>(time(nullptr))); 
    policy[1] = static_cast<float>(rand()) / RAND_MAX;

    srand(static_cast<unsigned int>(time(nullptr))); 
    policy[2] = static_cast<float>(rand()) / RAND_MAX;

    // normalize
    float sum = 0;
    sum = policy[0] + policy[1] + policy[2];

    for (unsigned long i=0; i<3; i++) {
        policy[i] = static_cast<float>(policy[i])/sum;
    }
}

// returns greedy action according to policy.
int IGA::greedy(Strategy strat) {
    srand(static_cast<unsigned int>(time(nullptr))); 
    float random = static_cast<float>(rand()) / RAND_MAX;

    // choose action based on policy probabilities.
    float p_rock = strat[0];
    float p_paper = strat[1];

    if (random < p_rock) {
        int action = 0; // rock.
        return action;
    }
    else if (random > p_rock && random < p_rock + p_paper) {
        int action = 1; // paper.
        return action;
    }
    else {
        int action = 2; // scissors.
        return action;
    }
}

std::tuple<Action, Strategy, Reward> IGA::act() {
    int action = greedy(policy);
    return {action, policy, NAN};
}

void IGA::observe(Reward r, Strategy x, Action action_y, Strategy true_y) {
    // silence unused parameters.
    (void)r;
    (void)x;
    (void)action_y;

    // retrieve reward matrix and the opponent action.
    RockPaperScissors RPS;

    // derivative for probability of action 0:
    float r_00 = RPS.step(0, 0)[0];
    float r_01 = RPS.step(0, 1)[0];
    float r_02 = RPS.step(0, 2)[0];
    float d_p0 = r_00*true_y[0] + r_01*true_y[1] + r_02*true_y[2];
    // derivative for probability of action 1:
    float r_10 = RPS.step(1, 0)[0];
    float r_11 = RPS.step(1, 1)[0];
    float r_12 = RPS.step(1, 2)[0];
    float d_p1 = r_10*true_y[0] + r_11*true_y[1] + r_12*true_y[2];
    // derivative for probability of action 2:
    float r_20 = RPS.step(2, 0)[0];
    float r_21 = RPS.step(2, 1)[0];
    float r_22 = RPS.step(2, 2)[0];
    float d_p2 = r_20*true_y[0] + r_21*true_y[1] + r_22*true_y[2];
    
    // update policy
    policy[0] += step_size * d_p0;
    policy[1] += step_size * d_p1;
    policy[2] += step_size * d_p2;

    // probability boundaries
    for (unsigned long i = 0; i < 3; i++) {
        if (policy[i] < 0.0) { policy[i] = 0.0; }
        if (policy[i] > 1.0) { policy[i] = 1.0; }
    }

    // normalize policy
    float sum = 0;
    sum = policy[0] + policy[1] + policy[2];

    for (unsigned long i=0; i<3; i++) {
        policy[i] = static_cast<float>(policy[i])/sum;
    }
}