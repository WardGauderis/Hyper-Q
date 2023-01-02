#include "PHC.h"
#include <algorithm>
#include <numeric>

// http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf

PHC::PHC(float alpha_, float delta_, float gamma_, float epsilon_) {
    alpha = alpha_;
    delta = delta_;
    gamma = gamma_;
    epsilon = epsilon_;
    current_state = 0;
    current_action = 0;

    for (int state=0; state<9; state++) {
        for (int action=0; action<3; action++) {
            if (action==0) {
                q_table[state][action] = 1;
            } else {
                q_table[state][action] = 0;
            }
        }
    }

    for (int state=0; state<9; state++) {
        policy_table[state] = {1,0,0};
    }
}

// returns greedy action according to policy.
unsigned long PHC::greedy() {
    srand(static_cast<unsigned int>(time(nullptr))); 
    float random = static_cast<float>(rand()) / RAND_MAX;

    // choose action based on policy probabilities.
    float p_rock = policy_table[current_state][0];
    float p_paper = policy_table[current_state][1];

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

std::pair<Action, Strategy> PHC::act() {
    // exploration factor.
    auto random = static_cast<float>(rand()) / RAND_MAX;
    if (random < epsilon) {
        srand(static_cast<unsigned int>(time(nullptr))); 
        auto action = static_cast<unsigned long>(rand() % 3);
        current_action = action;
        return {action, policy_table[current_state]};
    }
    // greedy.
    else {
        auto action = greedy();
        return {action, policy_table[current_state]};
    }
}

void PHC::observe(Reward r, Strategy x, Action action_y, Strategy true_y) {
    // silence unused parameters.
    (void)x;
    (void)true_y;

    // get next state.
    unsigned long next_state = 3*current_action + action_y;

    // get the q value for current state, and the max q value for the next state.
    float q = q_table[current_state][current_action];
    float q_max_next = *std::max_element(q_table[next_state], q_table[next_state]+3);
    
    // update q table.
    q_table[current_state][current_action] = (1-alpha)*q + alpha*(r + (gamma * q_max_next));
    
    // update policy table.
    float update = 0;
    if (current_action == static_cast<unsigned long>(std::distance(q_table[current_state], std::max_element(q_table[current_state], q_table[current_state]+3)))) {
        // i.e., if the current action equals argmax(Q[current_state]), then,
        update = delta;
    }
    else {
        update = (-delta)/(3-1);
    }
    policy_table[current_state][current_action] += update;
    
    // normalize new policy values.
    float sum = 0;
    for (unsigned long state=0; state < 9; state++) {
        sum = policy_table[state][0] +  policy_table[state][1] +  policy_table[state][2];

        for (unsigned long i=0; i<3; i++) {
            policy_table[state][i] = static_cast<float>(policy_table[state][i])/sum;
        }
    }

    // update current state.
    current_state = next_state;
}