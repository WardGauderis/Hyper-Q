//
// Created by ward on 12/30/22.
//

#ifndef HYPER_Q_OMNISCIENT_H
#define HYPER_Q_OMNISCIENT_H

#include "StrategyEstimation.h"

class Omniscient : public StrategyEstimation {
public:
    void observe(Action action_y, Strategy true_y) override {
        (void)action_y;
        strategy = true_y;
    }

    Strategy estimate() override {
        return strategy;
    }

private:
    Strategy strategy = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
};


#endif //HYPER_Q_OMNISCIENT_H
