//
// Created by ward on 12/30/22.
//

#ifndef HYPER_Q_BAYESIAN_H
#define HYPER_Q_BAYESIAN_H

#include "StrategyEstimation.h"

class Bayesian: public StrategyEstimation{
public:
    void observe(Action action_y, Strategy y) override {

    }

    Strategy estimate() override {
        return {};
    }
private:

};


#endif //HYPER_Q_BAYESIAN_H
