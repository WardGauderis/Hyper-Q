//
// Created by ward on 1/1/23.
//

#include <numeric>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include "Agent.h"

std::tuple<Action, Strategy, Reward> Agent::random_restart() {
    Strategy strategy;
    for (auto &i: strategy) {
        i = static_cast<double >(rand()) / static_cast<double>(RAND_MAX);
    }
    auto sum = std::accumulate(strategy.begin(), strategy.end(), 0.0);
    for (auto &i: strategy) {
        i /= sum;
    }

    assert(std::abs(std::accumulate(strategy.begin(), strategy.end(), 0.0) - 1.0) < 1e-6);
    
    return {strategy_to_action(strategy), strategy, NAN};
}

Action Agent::strategy_to_action(Strategy strategy) {
    auto random = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);

    for (unsigned int i = 0; i < strategy.size(); i++) {
        if (random < strategy[i]) {
            return i;
        }
        random -= strategy[i];
    }

    return strategy.size() - 1;
}

Strategy Agent::index_to_strategy(StrategyIndex index) {
    assert(index < num_strategies);
    auto x = index / grid_size;
    auto y = index % grid_size;
    assert(grid_size > x + y);
    assert(x < grid_size);
    assert(y < grid_size);

    std::array<unsigned int, 3> indices = {x, y, grid_size - 1 - x - y};

    Strategy strategy;
    const auto sum = grid_size - 1;

    for (size_t i = 0; i < indices.size(); ++i) {
        strategy[i] = static_cast<double>(indices[i]) / sum;
    }

    assert(std::abs(std::accumulate(strategy.begin(), strategy.end(), 0.0) - 1.0) < 1e-6);

    return strategy;
}

StrategyIndex Agent::strategy_to_index(Strategy strategy) {
    assert(std::abs(std::accumulate(strategy.begin(), strategy.end(), 0.0) - 1.0) < 1e-6);

    const auto sum = grid_size - 1;
    std::array<unsigned int, 3> indices{};

    indices[0] = static_cast<unsigned int>(std::round(strategy[0] * sum));
    indices[1] = static_cast<unsigned int>(std::round(strategy[1] * sum));
    indices[2] = sum - indices[0] - indices[1];

    assert(sum == indices[0] + indices[1] + indices[2]);
    assert(indices[0] < grid_size);
    assert(indices[1] < grid_size);
    assert(indices[2] < grid_size);

    return indices[0] * grid_size + indices[1];
}

StrategiesIndex Agent::strategies_to_index(Strategies strategies) {
    assert(strategies[0] < num_strategies);
    assert(strategies[1] < num_strategies);
    return strategies[0] * num_strategies + strategies[1];
}

bool Agent::valid_index(StrategyIndex index) {
    auto x = index / grid_size;
    auto y = index % grid_size;

    return grid_size > x + y;
}