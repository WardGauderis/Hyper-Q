//
// Created by ward on 12/31/22.
//

#ifndef HYPER_Q_MONOTONE_H
#define HYPER_Q_MONOTONE_H

#include <cmath>
#include "Agent.h"

class Monotone : public Agent {
public:
    explicit Monotone(const Strategy &strategy) : strategy(strategy) {}

    std::tuple<Action, Strategy, Reward> act() override {

        return {strategy_to_action(strategy), strategy, NAN};
    }

    double observe(Reward r, Action action_x, Strategy x, Action action_y, Strategy y) override {
        (void)r;
        (void)x;
        (void)action_y;
        (void)y;
        return NAN;
    }

private:
    Strategy strategy;

};

#endif //HYPER_Q_MONOTONE_H
