//
// Created by ward on 12/28/22.
//

#ifndef HYPER_Q_AGENT_H
#define HYPER_Q_AGENT_H

#include "Definitions.h"
#include <cstdlib>
#include <numeric>

class Agent {
public:
    virtual std::pair<Action, Strategy> act() = 0;

    virtual void observe(Reward r, Strategy x, Action action_y, Strategy y_true) = 0;

    virtual ~Agent() = default;

    static std::pair<Action, Strategy> random_restart() {
        Strategy strategy;
        for (auto &i: strategy) {
            i = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
        auto sum = std::accumulate(strategy.begin(), strategy.end(), 0.0f);
        for (auto &i: strategy) {
            i /= sum;
        }
        return {strategy_to_action(strategy), strategy};
    }

protected:
    static Action strategy_to_action(Strategy strategy) {
        auto random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        for (unsigned int i = 0; i < strategy.size(); i++) {
            if (random < strategy[i]) {
                return i;
            }
            random -= strategy[i];
        }

        return strategy.size() - 1;
    }
};


#endif //HYPER_Q_AGENT_H
