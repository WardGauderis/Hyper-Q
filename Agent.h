//
// Created by ward on 12/28/22.
//

#ifndef HYPER_Q_AGENT_H
#define HYPER_Q_AGENT_H

#include "Definitions.h"

class Agent {
public:
    virtual std::pair<Action , Strategy> act() = 0;

    virtual void observe(Reward r, Strategy x, Action action_y, Strategy y) = 0;

    virtual ~Agent() = default;
};


#endif //HYPER_Q_AGENT_H
