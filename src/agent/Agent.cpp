//
// Created by ward on 1/1/23.
//

#include <numeric>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include "Agent.h"
#include "Monotone.h"
#include "BayesianHyperQ.h"
#include "HyperQ.h"
#include "BayesianUltraQ.h"
#include "IGA.h"
#include "PHC.h"
#include "../estimation/Omniscient.h"
#include "../estimation/EMA.h"

Strategy Agent::random_strategy() {
    Strategy strategy;
    for (auto &i: strategy) {
        i = static_cast<double >(rand()) / static_cast<double>(RAND_MAX);
    }
    auto sum = std::accumulate(strategy.begin(), strategy.end(), 0.0);
    for (auto &i: strategy) {
        i /= sum;
    }

    assert(std::abs(std::accumulate(strategy.begin(), strategy.end(), 0.0) - 1.0) < 1e-6);
    return strategy;
}

std::tuple<Action, Strategy, Reward> Agent::random_restart() {
    auto strategy = random_strategy();
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
    assert(strategy[0] >= 0 && strategy[0] <= 1);
    assert(strategy[1] >= 0 && strategy[1] <= 1);
    assert(strategy[2] >= 0 && strategy[2] <= 1);

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

std::unique_ptr<Agent> Agent::from_json(const json &j) {
    std::string type = j.at("type");
    if (type == "monotone") {
        Strategy strategy = j.at("strategy");
        return std::make_unique<Monotone>(strategy);
    } else if (type == "omniscient_hyper_q") {
        auto estimation = std::make_unique<Omniscient>();
        double alpha = j.at("alpha");
        double gamma = j.at("gamma");
        std::optional<double> init;
        auto init_j = j.at("init");
        if (init_j.is_number()) {
            init = init_j;
        } else if (init_j != "random") {
            throw std::invalid_argument("Unknown init value: " + std::string(init_j));
        }
        return std::make_unique<HyperQ>(std::move(estimation), alpha, gamma, init);
    } else if (type == "ema_hyper_q") {
        double mu = j.at("mu");
        auto estimation = std::make_unique<EMA>(mu);
        double alpha = j.at("alpha");
        double gamma = j.at("gamma");
        std::optional<double> init;
        auto init_j = j.at("init");
        if (init_j.is_number()) {
            init = init_j;
        } else if (init_j != "random") {
            throw std::invalid_argument("Unknown init value: " + std::string(init_j));
        }
        return std::make_unique<HyperQ>(std::move(estimation), alpha, gamma, init);
    } else if (type == "bayesian_hyper_q") {
        double alpha = j.at("alpha");
        double gamma = j.at("gamma");
        double mu = j.at("mu");
        std::optional<double> init;
        auto init_j = j.at("init");
        if (init_j.is_number()) {
            init = init_j;
        } else if (init_j != "random") {
            throw std::invalid_argument("Unknown init value: " + std::string(init_j));
        }
        return std::make_unique<BayesianHyperQ>(alpha, gamma, mu, init);
    } else if (type == "bayesian_ultra_q") {
        double alpha = j.at("alpha");
        double gamma = j.at("gamma");
        double mu = j.at("mu");
        Similarity similarity = j.at("similarity");
        std::optional<double> init;
        auto init_j = j.at("init");
        if (init_j.is_number()) {
            init = init_j;
        } else if (init_j != "random") {
            throw std::invalid_argument("Unknown init value: " + std::string(init_j));
        }
        return std::make_unique<BayesianUltraQ>(alpha, gamma, mu, similarity, init);
    } else if (type == "iga") {
        double step_size = j.at("step_size");
        return std::make_unique<IGA>(step_size);
    } else if (type == "phc") {
        double alpha = j.at("alpha");
        double delta = j.at("delta");
        double gamma = j.at("gamma");
        std::optional<double> init;
        auto init_j = j.at("init");
        if (init_j.is_number()) {
            init = init_j;
        } else if (init_j != "random") {
            throw std::invalid_argument("Unknown init value: " + std::string(init_j));
        }
        return std::make_unique<PHC>(alpha, delta, gamma, init);
    } else {
        throw std::invalid_argument("Unknown agent type: " + type);
    }
}
