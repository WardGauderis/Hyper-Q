//
// Created by ward on 1/3/23.
//

#ifndef HYPER_Q_BAYESIANULTRAQ_H
#define HYPER_Q_BAYESIANULTRAQ_H

#include "BayesianHyperQ.h"

class BayesianUltraQ : public BayesianHyperQ {
public:

    BayesianUltraQ(double alpha, double gamma, double mu) : BayesianHyperQ(alpha, gamma, mu) {}

    void observe(Reward r, Action action_x, Strategy _x, Action action_y, Strategy _y) override {
        update_posterior(action_y);

        auto max = greedy().second;

        for (StrategyIndex x_index = 0; x_index < num_strategies; ++x_index) {
            if (!valid_index(x_index)) continue;
            auto x = index_to_strategy(x_index);

            for (StrategyIndex y_index = 0; y_index < num_strategies; ++y_index) {
                if (!valid_index(y_index)) continue;
                auto index = strategies_to_index({x_index, y_index});

                hyper_q_table[index] += alpha * posterior_table[y_index] * x[action_x] * (r + gamma * max - hyper_q_table[index]);
            }
        }
    }
};


#endif //HYPER_Q_BAYESIANULTRAQ_H
