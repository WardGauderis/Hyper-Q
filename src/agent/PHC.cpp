#include "PHC.h"
#include <algorithm>
#include <cmath>

// http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf

PHC::PHC(float alpha_, float delta_, float gamma_, std::optional<double> init) {
    alpha = static_cast<double>(alpha_);
    delta = static_cast<double>(delta_);
    gamma = static_cast<double>(gamma_);
    current_state = 0;

    // q table creation.
    if (init.has_value()) {
        for (auto &q: q_table) {
            for (auto &q_value: q) {
                q_value = init.value();
            }
        }
    } else {
        for (auto &q: q_table) {
            for (auto &q_value: q) {
                q_value = static_cast<double>(rand()) / RAND_MAX;
            }
        }
    }

    // policy table creation.
    for (int state = 0; state < 9; state++) {
        // probabilities
        double p1 = static_cast<double>(rand()) / RAND_MAX;
        double p2 = static_cast<double>(rand()) / RAND_MAX;
        double p3 = static_cast<double>(rand()) / RAND_MAX;

        double sum = p1 + p2 + p3;

        policy_table[state] = {p1 / sum, p2 / sum, p3 / sum};
    }
}


std::tuple<Action, Strategy, Reward> PHC::act() {
    auto action = Agent::strategy_to_action(policy_table[current_state]);
    return {action, policy_table[current_state], NAN};
}

double PHC::observe(Reward r, Action action_x, Strategy x, Action action_y, Strategy true_y) {
    // silence unused parameters.
    (void) x;
    (void) true_y;

    // get next state.
    unsigned long next_state = 3 * action_x + action_y;

    // get the q value for current state, and the max q value for the next state.
    double q = q_table[current_state][action_x];
    double q_max_next = *std::max_element(q_table[next_state], q_table[next_state] + 3);

    // update q table.
    q_table[current_state][action_x] = (1 - alpha) * q + alpha * (r + (gamma * q_max_next));

    // update policy table.
    double update = 0;
    if (action_x == static_cast<unsigned long>(std::distance(q_table[current_state],
                                                                   std::max_element(q_table[current_state],
                                                                                    q_table[current_state] + 3)))) {
        // i.e., if the current action equals argmax(Q[current_state]), then,
        update = delta;
    } else {
        update = (-delta) / (3 - 1);
    }
    policy_table[current_state][action_x] += update;

    // probability boundaries
    for (unsigned long state = 0; state < 9; state++) {
        for (unsigned long i = 0; i < 3; i++) {
            if (policy_table[state][i] < 0.0) { policy_table[state][i] = 0.0; }
            if (policy_table[state][i] > 1.0) { policy_table[state][i] = 1.0; }
        }
    }

    // normalize new policy values.
    double sum = 0;
    for (unsigned long state = 0; state < 9; state++) {
        sum = policy_table[state][0] + policy_table[state][1] + policy_table[state][2];

        for (unsigned long i = 0; i < 3; i++) {
            policy_table[state][i] = static_cast<double>(policy_table[state][i] / sum);
        }
    }

    // update current state.
    current_state = next_state;

    return NAN;
}


//#include "PHC.h"
//#include <algorithm>
//#include <cmath>
//
//// http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf
//
//PHC::PHC(float alpha_, float delta_, float gamma_, std::optional<double> init) {
//    alpha = static_cast<double>(alpha_);
//    delta = static_cast<double>(delta_);
//    gamma = static_cast<double>(gamma_);
//    current_state = 0;
//    current_action = 0;
//
//    // q table creation.
//    if (init.has_value()) {
//        for (auto &q: q_table) {
//            for (auto &q_value: q) {
//                q_value = init.value();
//            }
//        }
//    } else {
//        for (auto &q: q_table) {
//            for (auto &q_value: q) {
//                q_value = static_cast<double>(rand()) / RAND_MAX;
//            }
//        }
//    }
//
//    // policy table creation.
//    for (int state = 0; state < 9; state++) {
//        // probabilities
//        double p1 = static_cast<double>(rand()) / RAND_MAX;
//        double p2 = static_cast<double>(rand()) / RAND_MAX;
//        double p3 = static_cast<double>(rand()) / RAND_MAX;
//
//        double sum = p1 + p2 + p3;
//
//        policy_table[state] = {p1 / sum, p2 / sum, p3 / sum};
//    }
//}
//
//
//std::tuple<Action, Strategy, Reward> PHC::act() {
//    auto action = Agent::strategy_to_action(policy_table[current_state]);
//    return {action, policy_table[current_state], NAN};
//}
//
//double PHC::observe(Reward r, Action action_x, Strategy x, Action action_y, Strategy true_y) {
//    // silence unused parameters.
//    (void) x;
//    (void) true_y;
//
//    // get next state.
//    unsigned long next_state = 3 * current_action + action_y;
//
//    // get the q value for current state, and the max q value for the next state.
//    double q = q_table[current_state][current_action];
//    double q_max_next = *std::max_element(q_table[next_state], q_table[next_state] + 3);
//
//    // update q table.
//    q_table[current_state][current_action] = (1 - alpha) * q + alpha * (r + (gamma * q_max_next));
//
//    // update policy table.
//    double update = 0;
//    if (current_action == static_cast<unsigned long>(std::distance(q_table[current_state],
//                                                                   std::max_element(q_table[current_state],
//                                                                                    q_table[current_state] + 3)))) {
//        // i.e., if the current action equals argmax(Q[current_state]), then,
//        update = delta;
//    } else {
//        update = (-delta) / (3 - 1);
//    }
//    policy_table[current_state][current_action] += update;
//
//    // probability boundaries
//    for (unsigned long state = 0; state < 9; state++) {
//        for (unsigned long i = 0; i < 3; i++) {
//            if (policy_table[state][i] < 0.0) { policy_table[state][i] = 0.0; }
//            if (policy_table[state][i] > 1.0) { policy_table[state][i] = 1.0; }
//        }
//    }
//
//    // normalize new policy values.
//    double sum = 0;
//    for (unsigned long state = 0; state < 9; state++) {
//        sum = policy_table[state][0] + policy_table[state][1] + policy_table[state][2];
//
//        for (unsigned long i = 0; i < 3; i++) {
//            policy_table[state][i] = static_cast<double>(policy_table[state][i] / sum);
//        }
//    }
//
//    // update current state.
//    current_state = next_state;
//
//    return NAN;
//}