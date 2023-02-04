//
// Created by ward on 2/4/23.
//

#ifndef HYPER_Q_EXPLORATIONMETHOD_H
#define HYPER_Q_EXPLORATIONMETHOD_H

#include "../agent/Agent.h"

struct Returns {
    Action action_x, action_y;
    Strategy strategy_x, strategy_y;
    Reward value_x, value_y;
};

class ExplorationMethod {
public:
    virtual Returns
    act(const std::unique_ptr<Agent> &agent_x, const std::unique_ptr<Agent> &agent_y, unsigned int iteration) = 0;

    static std::unique_ptr<ExplorationMethod> from_json(const json &j);

    virtual ~ExplorationMethod() = default;
};

class RandomRestart : public ExplorationMethod {
public:
    RandomRestart(unsigned int every) : every(every) {}

    Returns
    act(const std::unique_ptr<Agent> &agent_x, const std::unique_ptr<Agent> &agent_y, unsigned int iteration) override;

private:
    unsigned int every;
};

class EpsilonGreedy : public ExplorationMethod {
public:
    EpsilonGreedy(double epsilon) : epsilon(epsilon) {}

    Returns
    act(const std::unique_ptr<Agent> &agent_x, const std::unique_ptr<Agent> &agent_y, unsigned int iteration) override;

private:
    double epsilon;
};

class EpsilonGreedyDecay : public ExplorationMethod {
public:
    Returns
    act(const std::unique_ptr<Agent> &agent_x, const std::unique_ptr<Agent> &agent_y, unsigned int iteration) override;
};


#endif //HYPER_Q_EXPLORATIONMETHOD_H
