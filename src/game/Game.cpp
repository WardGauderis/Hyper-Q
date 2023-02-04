//
// Created by ward on 2/4/23.
//

#include "Game.h"
#include "RockPaperScissors.h"
#include "HillClimbing.h"


std::unique_ptr<Game> Game::from_json(const json &j) {
    std::string game = j;
    if (game == "rock_paper_scissors")
        return std::make_unique<RockPaperScissors>();
    else if (game == "hill_climbing")
        return std::make_unique<HillClimbing>();
    else
        throw std::runtime_error("Unknown game: " + game);
}

