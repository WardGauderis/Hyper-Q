//
// Created by ward on 12/30/22.
//

#ifndef HYPER_Q_EMA_H
#define HYPER_Q_EMA_H

#include "StrategyEstimation.h"
#include "../agent/Agent.h"


class EMA : public StrategyEstimation {
public:
    EMA(double mu) : mu(mu) {}

    void observe(Action action_y, Strategy true_y) override {
        (void)true_y;
        for (unsigned int i = 0; i < strategy.size(); i++) {
            strategy[i] = (1 - mu) * strategy[i] + mu * static_cast<double >(action_y == i);
        }
    }

    Strategy estimate() override {
        return strategy;
    }

    void random_restart() override {
        strategy = Agent::random_strategy();
    }

private:
    double mu;
    Strategy strategy = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
};


#endif //HYPER_Q_EMA_H
