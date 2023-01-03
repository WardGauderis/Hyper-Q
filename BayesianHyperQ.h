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
        std::fill(hyper_q_table.begin(), hyper_q_table.end(), 0);
        std::fill(posterior_table.begin(), posterior_table.end(), 0);
    }

    std::pair<Action, Strategy> act() {
        auto [x, value] = greedy();
        log_file << value << std::endl;
        return {strategy_to_action(x), x};
    }

    void observe(Reward r, Strategy x, Action action_y, Strategy true_y) {
        (void)true_y;
        auto x_index = strategy_to_index(x);

        update_posterior(action_y);

        auto max = greedy().second;
        for (StrategyIndex y_index = 0; y_index < num_strategies; ++y_index) {
            if (!valid_index(y_index)) continue;
            auto index = strategies_to_index({x_index, y_index});

            hyper_q_table[index] += alpha * posterior_table[y_index] * (r + gamma * max - hyper_q_table[index]);
        }
    }

private:
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
