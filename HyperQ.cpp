//
// Created by ward on 12/28/22.
//

#include "HyperQ.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <cmath>

HyperQ::HyperQ(std::unique_ptr<StrategyEstimation> estimation, float alpha, float gamma) : estimation(
        std::move(estimation)), alpha(alpha), gamma(gamma) {
    std::fill(hyper_q_table.begin(), hyper_q_table.end(), 0);
}


std::pair<Action, Strategy> HyperQ::act() {
    auto x = greedy().first;
    return {strategy_to_action(x), x};
}

void HyperQ::observe(Reward r, Strategy x, Action action_y, Strategy true_y) {
    auto y = estimation->estimate();

    auto x_index = strategy_to_index(x);
    auto y_index = strategy_to_index(y);

    auto index = strategies_to_index({x_index, y_index});

    estimation->observe(action_y, true_y);

    auto max = greedy().second;
    hyper_q_table[index] += alpha * (r + gamma * max - hyper_q_table[index]);
}

std::pair<Strategy, Reward> HyperQ::greedy() {
    auto y = estimation->estimate();
    auto y_index = strategy_to_index(y);

    auto max = std::numeric_limits<float>::lowest();
    StrategyIndex max_x = 0;

    for (StrategyIndex x = 0; x < num_strategies; ++x) {
        if (!valid_index(x)) continue;

        const auto index = strategies_to_index({x, y_index});
        if (hyper_q_table[index] > max) {
            max = hyper_q_table[index];
            max_x = x;
        }
    }

    return {index_to_strategy(max_x), max};
}

StrategyIndex HyperQ::strategy_to_index(Strategy strategy) {
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

Strategy HyperQ::index_to_strategy(StrategyIndex index) {

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
        strategy[i] = static_cast<float>(indices[i]) / sum;
    }

    return strategy;
}

StrategiesIndex HyperQ::strategies_to_index(Strategies strategies) {
    assert(strategies[0] < num_strategies);
    assert(strategies[1] < num_strategies);
    return strategies[0] * num_strategies + strategies[1];
}

bool HyperQ::valid_index(StrategyIndex index) {
    auto x = index / grid_size;
    auto y = index % grid_size;

    return grid_size > x + y;
}
