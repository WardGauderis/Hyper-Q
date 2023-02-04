//
// Created by ward on 12/28/22.
//

#ifndef HYPER_Q_ROCKPAPERSCISSORS_H
#define HYPER_Q_ROCKPAPERSCISSORS_H


#include "Game.h"

class RockPaperScissors : public Game {
public:
    Rewards step(Action action_x, Action action_y) override;
};


#endif //HYPER_Q_ROCKPAPERSCISSORS_H
