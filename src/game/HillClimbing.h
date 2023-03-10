#ifndef COOPERATION_GAME_H
#define COOPERATION_GAME_H

#include <array>
#include <cassert>
#include "../agent/Monotone.h"
#include "Game.h"

class HillClimbing : public Game {
public:
    HillClimbing();

    Rewards step(Action action_x, Action action_y) override;


private:
    std::array<std::array<double, 3>, 3> probability_matrix{};
    std::array<std::array<double, 3>, 3> reward_matrix1{};
    std::array<std::array<double, 3>, 3> reward_matrix2{};

    static std::tuple<std::array<std::array<double, 3>, 3>, std::array<std::array<double, 3>, 3>, std::array<std::array<double, 3>, 3>>
    initialize_game_rewards();

};

#endif