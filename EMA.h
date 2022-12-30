//
// Created by ward on 12/30/22.
//

#ifndef HYPER_Q_EMA_H
#define HYPER_Q_EMA_H

#include "StrategyEstimation.h"


class EMA : public StrategyEstimation {
public:
    EMA(float mu) : mu(mu) {}

    void observe(Action action_y, Strategy true_y) override {
        for (unsigned int i = 0; i < strategy.size(); i++) {
            strategy[i] = (1 - mu) * strategy[i] + mu * static_cast<float>(action_y == i);
        }
    }

    Strategy estimate() override {
        return strategy;
    }

private:
    float mu;
    Strategy strategy = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
};


#endif //HYPER_Q_EMA_H
