//
// Created by ward on 12/28/22.
//

#ifndef HYPER_Q_HYPERQ_H
#define HYPER_Q_HYPERQ_H

#include "Agent.h"
#include "StrategyEstimation.h"
#include "Definitions.h"
#include <memory>
#include <limits>


class HyperQ : public Agent {
public:
    HyperQ(std::unique_ptr<StrategyEstimation> estimation, double alpha, double gamma) : estimation(
            std::move(estimation)), alpha(alpha), gamma(gamma) {
        for (auto &q: hyper_q_table) {
            q = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
        }
    }

    std::tuple<Action, Strategy, Reward> act() override {
        auto [x, value] = greedy();
        return {strategy_to_action(x), x, value};
    }

    void observe(Reward r, Strategy x, Action action_y, Strategy true_y) override {
        auto y = estimation->estimate();

        auto x_index = strategy_to_index(x);
        auto y_index = strategy_to_index(y);

        auto index = strategies_to_index({x_index, y_index});

        estimation->observe(action_y, true_y);

        auto max = greedy().second;
        hyper_q_table[index] += alpha * (r + gamma * max - hyper_q_table[index]);
    }

    std::tuple<Action, Strategy, Reward> random_restart() override {
        auto [action_x, x, reward] = Agent::random_restart();
        auto y = estimation->estimate();
        auto y_index = strategy_to_index(y);
        auto x_index = strategy_to_index(x);
        auto index = strategies_to_index({x_index, y_index});
        return {action_x, x, hyper_q_table[index]};
    }

private:
    std::array<Reward, num_pairs> hyper_q_table{};
    std::unique_ptr<StrategyEstimation> estimation;
    double alpha;
    double gamma;

    std::pair<Strategy, Reward> greedy() {
        auto y = estimation->estimate();
        auto y_index = strategy_to_index(y);

        auto max = std::numeric_limits<double>::lowest();
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
};

#endif //HYPER_Q_HYPERQ_H

