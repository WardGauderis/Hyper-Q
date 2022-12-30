//
// Created by ward on 12/28/22.
//

#ifndef HYPER_Q_HYPERQ_H
#define HYPER_Q_HYPERQ_H

#include "Agent.h"
#include "StrategyEstimation.h"
#include "Definitions.h"
#include <memory>

static const unsigned int grid_size = 5;
static const unsigned int num_strategies = grid_size * grid_size;
static const unsigned int num_pairs = num_strategies * num_strategies;


class HyperQ : public Agent {
public:
    HyperQ(std::unique_ptr<StrategyEstimation> estimation, float alpha, float gamma);

    std::pair<Action, Strategy> act() override;

    void observe(Reward r, Strategy x, Action action_y, Strategy y) override;
private:
    std::array<Reward, num_pairs> hyper_q_table;
    std::unique_ptr<StrategyEstimation> estimation;
    float alpha;
    float gamma;

    std::pair<Strategy , Reward> greedy();

    static Strategy index_to_strategy(StrategyIndex index);

    static StrategyIndex strategy_to_index(Strategy strategy);

    static StrategiesIndex strategies_to_index(Strategies strategies);

    bool valid_index(StrategyIndex index);
};

#endif //HYPER_Q_HYPERQ_H

