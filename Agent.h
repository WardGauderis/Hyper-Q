//
// Created by ward on 12/28/22.
//

#ifndef HYPER_Q_AGENT_H
#define HYPER_Q_AGENT_H

#include "Definitions.h"

static const unsigned int grid_size = 25;
static const unsigned int num_strategies = grid_size * grid_size;
static const unsigned int num_pairs = num_strategies * num_strategies;

class Agent {
public:
    virtual std::pair<Action, Strategy> act() = 0;

    virtual void observe(Reward r, Strategy x, Action action_y, Strategy y_true) = 0;

    virtual ~Agent() = default;

    static std::pair<Action, Strategy> random_restart();

protected:
    static Action strategy_to_action(Strategy strategy);

    static Strategy index_to_strategy(StrategyIndex index);

    static StrategyIndex strategy_to_index(Strategy strategy);

    static StrategiesIndex strategies_to_index(Strategies strategies);

    static bool valid_index(StrategyIndex index);
};


#endif //HYPER_Q_AGENT_H
