//
// Created by ward on 1/3/23.
//

#ifndef HYPER_Q_BAYESIANULTRAQ_H
#define HYPER_Q_BAYESIANULTRAQ_H

#include "BayesianHyperQ.h"

class BayesianUltraQ : public BayesianHyperQ {
public:

    BayesianUltraQ(double alpha, double gamma, double mu) : BayesianHyperQ(alpha, gamma, mu) {}

    double observe(Reward r, Action action_x, Strategy original_x, Action action_y, Strategy _y) override {
        update_posterior(action_y);

        auto max = greedy().second;
        auto sum = 0.0;
        for (StrategyIndex x_index = 0; x_index < num_strategies; ++x_index) {
            if (!valid_index(x_index)) continue;
            auto x = index_to_strategy(x_index);

            for (StrategyIndex y_index = 0; y_index < num_strategies; ++y_index) {
                if (!valid_index(y_index)) continue;
                auto index = strategies_to_index({x_index, y_index});

                auto similarity = 0.0;
                for (unsigned int i = 0; i < x.size(); ++i) {
                    similarity += x[i] * original_x[i];
                }

                auto bellman_error = similarity * posterior_table[y_index] * (r + gamma * max - hyper_q_table[index]);
                hyper_q_table[index] += alpha * bellman_error;
                sum += bellman_error;
            }
        }

        return sum;
    }
};


#endif //HYPER_Q_BAYESIANULTRAQ_H
