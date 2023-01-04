#include "PHC.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

// http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf

PHC::PHC(float alpha_, float delta_, float gamma_, float epsilon_) {
    alpha = static_cast<double>(alpha_);
    delta = static_cast<double>(delta_);
    gamma = static_cast<double>(gamma_);
    epsilon = static_cast<double>(epsilon_);
    current_state = 0;
    current_action = 0;

    for (int state=0; state<9; state++) {
        for (int action=0; action<3; action++) {
            if (action==0) {
                q_table[state][action] = 0.1;
            } else {
                q_table[state][action] = 0.1;
            }
        }
    }

    for (int state=0; state<9; state++) {
        // probabilities 
        double p1 = static_cast<double>(rand()) / RAND_MAX;
        double p2 = static_cast<double>(rand()) / RAND_MAX;
        double p3 = static_cast<double>(rand()) / RAND_MAX;

        double sum = p1+p2+p3;

        policy_table[state] = {p1/sum, p2/sum, p3/sum};
    }
}

// returns greedy action according to policy.
unsigned long PHC::greedy() {
    auto random = static_cast<double>(rand()) / RAND_MAX;

    // choose action based on policy probabilities.
    double p_rock = policy_table[current_state][0];
    double p_paper = policy_table[current_state][1];

    if (random < p_rock) {
        unsigned long action = 0; // rock.
        return action;
    }
    else if (random > p_rock && random < p_rock + p_paper) {
        unsigned long action = 1; // paper.
        current_action = action;
        return action;
    }
    else {
        unsigned long action = 2; // scissors.
        current_action = action;
        return action;
    }
}

std::tuple<Action, Strategy, Reward> PHC::act() {
    // exploration factor.
    auto random = static_cast<double>(rand()) / RAND_MAX;
    if (random < epsilon) {
        auto action = static_cast<unsigned long>(rand() % 3);
        current_action = action;
        return {action, policy_table[current_state], NAN};
    }
    // greedy.
    else {
        auto action = greedy();
        return {action, policy_table[current_state], NAN};
    }
}

double PHC::observe(Reward r, Action action_x, Strategy x, Action action_y, Strategy true_y) {
    // silence unused parameters.
    (void)x;
    (void)true_y;

    // get next state.
    unsigned long next_state = 3*current_action + action_y;

    // get the q value for current state, and the max q value for the next state.
    double q = q_table[current_state][current_action];
    double q_max_next = *std::max_element(q_table[next_state], q_table[next_state]+3);
    
    // update q table.
    q_table[current_state][current_action] = (1-alpha)*q + alpha*(r + (gamma * q_max_next));
    
    // update policy table.
    double update = 0;
    if (current_action == static_cast<unsigned long>(std::distance(q_table[current_state], std::max_element(q_table[current_state], q_table[current_state]+3)))) {
        // i.e., if the current action equals argmax(Q[current_state]), then,
        update = delta;
    }
    else {
        update = (-delta)/(3-1);
    }
    policy_table[current_state][current_action] += update;
    
    // probability boundaries
    for (unsigned long state=0; state < 9; state++) {
        for (unsigned long i=0; i<3; i++) {
            if (policy_table[state][i] < 0.0) { policy_table[state][i] = 0.0; }
            if (policy_table[state][i] > 1.0) { policy_table[state][i] = 1.0; }
        }
    } 

    // normalize new policy values.
    double sum = 0;
    for (unsigned long state=0; state < 9; state++) {
        sum = policy_table[state][0] +  policy_table[state][1] +  policy_table[state][2];

        for (unsigned long i=0; i<3; i++) {
            policy_table[state][i] = static_cast<double>(policy_table[state][i]/sum);
        }
    }

    // update current state.
    current_state = next_state;

    return NAN;
}