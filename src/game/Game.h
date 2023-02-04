//
// Created by ward on 12/28/22.
//

#ifndef HYPER_Q_GAME_H
#define HYPER_Q_GAME_H

#include "../utils/Definitions.h"

#include "../utils/json.hpp"

using json = nlohmann::json;


class Game {
public:
    virtual Rewards step(Action action_x, Action action_y) = 0;

    virtual ~Game() = default;

    static std::unique_ptr<Game> from_json(const json &j);
};


#endif //HYPER_Q_GAME_H
