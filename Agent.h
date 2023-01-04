//
// Created by ward on 12/28/22.
//

#ifndef HYPER_Q_AGENT_H
#define HYPER_Q_AGENT_H

#include <string>
#include <fstream>
#include <tuple>
#include "Definitions.h"

class Agent {
public:
    virtual std::tuple<Action, Strategy, Reward> act() = 0;

    virtual double observe(Reward r, Action action_x, Strategy x, Action action_y, Strategy y) = 0;

    virtual ~Agent() = default;

    virtual std::tuple<Action, Strategy, Reward> random_restart();

    static Strategy random_strategy();

protected:
    static Action strategy_to_action(Strategy strategy);

    static Strategy index_to_strategy(StrategyIndex index);

    static StrategyIndex strategy_to_index(Strategy strategy);

    static StrategiesIndex strategies_to_index(Strategies strategies);

    static bool valid_index(StrategyIndex index);
};


#endif //HYPER_Q_AGENT_H
