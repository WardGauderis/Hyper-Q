//
// Created by ward on 2/4/23.
//

#include "ExplorationMethod.h"

std::unique_ptr<ExplorationMethod> ExplorationMethod::from_json(const json &j) {
    std::string type = j.at("type");
    if (type == "random_restart") {
        unsigned int every = j.at("every");
        return std::make_unique<RandomRestart>(every);
    } else if (type == "epsilon_greedy") {
        double epsilon = j.at("epsilon");
        return std::make_unique<EpsilonGreedy>(epsilon);
    } else if (type == "epsilon_greedy_decay") {
        return std::make_unique<EpsilonGreedyDecay>();
    } else {
        throw std::invalid_argument("Unknown exploration method: " + type);
    }
}

Returns
RandomRestart::act(const std::unique_ptr<Agent> &agent_x, const std::unique_ptr<Agent> &agent_y,
                   unsigned int iteration) {
    Returns returns{};
    if (iteration % every == 0) {
        std::tie(returns.action_x, returns.strategy_x, returns.value_x) = agent_x->random_restart();
        std::tie(returns.action_y, returns.strategy_y, returns.value_y) = agent_y->random_restart();
    } else {
        std::tie(returns.action_x, returns.strategy_x, returns.value_x) = agent_x->act();
        std::tie(returns.action_y, returns.strategy_y, returns.value_y) = agent_y->act();
    }
    return returns;
}

Returns
EpsilonGreedy::act(const std::unique_ptr<Agent> &agent_x, const std::unique_ptr<Agent> &agent_y,
                   unsigned int iteration) {
    double random = static_cast<double>(rand()) / RAND_MAX;
    Returns returns{};
    if (random < epsilon) {
        std::tie(returns.action_x, returns.strategy_x, returns.value_x) = agent_x->random_restart();
    } else {
        std::tie(returns.action_x, returns.strategy_x, returns.value_x) = agent_x->act();
    }
    random = static_cast<double>(rand()) / RAND_MAX;
    if (random < epsilon) {
        std::tie(returns.action_y, returns.strategy_y, returns.value_y) = agent_y->random_restart();
    } else {
        std::tie(returns.action_y, returns.strategy_y, returns.value_y) = agent_y->act();
    }
    return returns;
}

Returns EpsilonGreedyDecay::act(const std::unique_ptr<Agent> &agent_x, const std::unique_ptr<Agent> &agent_y,
                                unsigned int iteration) {
    double random = static_cast<double>(rand()) / RAND_MAX;
    Returns returns{};
    if (random < 1.0 / (1.0 + iteration)) {
        std::tie(returns.action_x, returns.strategy_x, returns.value_x) = agent_x->random_restart();
    } else {
        std::tie(returns.action_x, returns.strategy_x, returns.value_x) = agent_x->act();
    }
    random = static_cast<double>(rand()) / RAND_MAX;
    if (random < 1.0 / (1.0 + iteration)) {
        std::tie(returns.action_y, returns.strategy_y, returns.value_y) = agent_y->random_restart();
    } else {
        std::tie(returns.action_y, returns.strategy_y, returns.value_y) = agent_y->act();
    }
    return returns;
}
