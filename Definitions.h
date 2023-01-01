//
// Created by ward on 12/30/22.
//

#ifndef HYPER_Q_DEFINITIONS_H
#define HYPER_Q_DEFINITIONS_H

#include <array>

using Action = unsigned int;

using Reward = double;

using Rewards = std::array<Reward, 2>;

using Probability = double;

using Strategy = std::array<Probability, 3>;

using StrategyIndex = unsigned int;

using Strategies = std::array<StrategyIndex, 2>;

using StrategiesIndex = unsigned int;

#endif //HYPER_Q_DEFINITIONS_H
