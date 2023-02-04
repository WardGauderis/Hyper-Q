//
// Created by ward on 12/30/22.
//

#ifndef HYPER_Q_DEFINITIONS_H
#define HYPER_Q_DEFINITIONS_H

static const unsigned int grid_size = 25;
static const unsigned int num_strategies = grid_size * grid_size;
static const unsigned int num_pairs = num_strategies * num_strategies;

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
