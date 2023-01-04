//
// Created by ward on 1/1/23.
//

#ifndef HYPER_Q_BAYESIANHYPERQ_H
#define HYPER_Q_BAYESIANHYPERQ_H


#include "Agent.h"
#include <memory>
#include <limits>
#include <deque>
#include <complex>
#include <numeric>
#include <cassert>

class BayesianHyperQ : public Agent {
public:
    BayesianHyperQ(double alpha, double gamma, double mu) : alpha(alpha), gamma(gamma), mu(mu) {
        for (auto &posterior: posterior_table) {
            posterior = 1.0 / num_strategies;
        }
        hyper_q_table.fill(1);
    }

    std::tuple<Action, Strategy, Reward> act() override {
        auto [x, value] = greedy();
        return {strategy_to_action(x), x, value};
    }

    double observe(Reward r, Action action_x, Strategy x, Action action_y, Strategy true_y) override {
        (void) true_y;
        auto x_index = strategy_to_index(x);

        update_posterior(action_y);

        auto max = greedy().second;
        auto sum = 0.0;
        for (StrategyIndex y_index = 0; y_index < num_strategies; ++y_index) {
            if (!valid_index(y_index)) continue;
            auto index = strategies_to_index({x_index, y_index});

            auto bellman_error = posterior_table[y_index] * (r + gamma * max - hyper_q_table[index]);
            hyper_q_table[index] += alpha * bellman_error;
            sum += bellman_error;
        }

        return sum;
    }

    std::tuple<Action, Strategy, Reward> random_restart() override {
        history.clear();
        for (auto &posterior: posterior_table) {
            posterior = 1.0 / num_strategies;
        }

        auto [action_x, x, reward] = Agent::random_restart();
        auto value = 0.0;

        auto x_index = strategy_to_index(x);

        for (StrategyIndex y_index = 0; y_index < num_strategies; ++y_index) {
            if (!valid_index(y_index)) continue;
            auto index = strategies_to_index({x_index, y_index});
            value += hyper_q_table[index] * posterior_table[y_index];
        }

        return {action_x, x, value};
    }

protected:
    std::array<Reward, num_strategies> posterior_table{};
    std::array<Reward, num_pairs> hyper_q_table{};
    std::deque<Action> history{};

    double alpha;
    double gamma;
    double mu;

    std::pair<Strategy, Reward> greedy() {
        auto max = std::numeric_limits<double>::lowest();
        StrategyIndex max_x = 0;

        for (StrategyIndex x = 0; x < num_strategies; ++x) {
            if (!valid_index(x)) continue;

            auto sum = 0.0;
            for (StrategyIndex y = 0; y < num_strategies; ++y) {
                if (!valid_index(y)) continue;

                sum += hyper_q_table[strategies_to_index({x, y})] * posterior_table[y];
            }

            if (sum > max) {
                max = sum;
                max_x = x;
            }
        }

        return {index_to_strategy(max_x), max};
    }

    void update_posterior(Action action_y) {
        history.push_front(action_y);
        if (static_cast<double>(history.size()) > 1 / mu) history.pop_back();

        std::fill(posterior_table.begin(), posterior_table.end(), 0);

        auto sum = 0.0;

        for (StrategyIndex y_index = 0; y_index < num_strategies; ++y_index) {
            if (!valid_index(y_index)) continue;
            const auto y = index_to_strategy(y_index);

            auto log_p = 0.0;
            for (unsigned int k = 0; k < history.size(); ++k) {
                const auto action_k = history[k];
                const auto probability = y[action_k];

                log_p += std::log(probability) * (1 - mu * k);
            }

            log_p += std::log(1.0 / num_strategies);

            posterior_table[y_index] = std::exp(log_p);
            sum += posterior_table[y_index];
        }

        for (auto &p: posterior_table) p /= sum;
    }
};

#endif //HYPER_Q_BAYESIANHYPERQ_H
