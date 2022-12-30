//
// Created by ward on 12/30/22.
//

#ifndef HYPER_Q_OMNISCIENT_H
#define HYPER_Q_OMNISCIENT_H

#include "StrategyEstimation.h"

class Omniscient : public StrategyEstimation {
public:
    void observe(Action action_y, Strategy true_y) override {
        strategy = true_y;
    }

    Strategy estimate() override {
        return strategy;
    }

private:
    Strategy strategy = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
};


#endif //HYPER_Q_OMNISCIENT_H
