//
// Created by ward on 12/30/22.
//

#ifndef HYPER_Q_STRATEGYESTIMATION_H
#define HYPER_Q_STRATEGYESTIMATION_H

#include <iostream>
#include "../utils/Definitions.h"
#include "../utils/json.hpp"

using json = nlohmann::json;

class StrategyEstimation {
public:
    virtual void observe(Action action_y, Strategy true_y) = 0;

    virtual Strategy estimate() = 0;

    virtual void random_restart() = 0;

    virtual ~StrategyEstimation() = default;

    static std::unique_ptr<StrategyEstimation> from_json(const json &j);
};


#endif //HYPER_Q_STRATEGYESTIMATION_H
