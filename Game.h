//
// Created by ward on 12/28/22.
//

#ifndef HYPER_Q_GAME_H
#define HYPER_Q_GAME_H

#include "Definitions.h"

class Game {
public:
    virtual Rewards step(Action action_x, Action action_y) = 0;

    virtual ~Game() = default;
};


#endif //HYPER_Q_GAME_H
