//
// Created by ward on 12/28/22.
//

#include <stdexcept>
#include "RockPaperScissors.h"

/**
 * @brief RockPaperScissors::step
 * @param action_x 0 = rock, 1 = paper, 2 = scissors
 * @param action_y 0 = rock, 1 = paper, 2 = scissors
 * @return reward for player a and b
 */
Rewards RockPaperScissors::step(Action action_x, Action action_y) {
    if (action_x == action_y) {
        return {0, 0};
    } else if (action_x == 0 && action_y == 1) {
        return {-1, 1};
    } else if (action_x == 0 && action_y == 2) {
        return {1, -1};
    } else if (action_x == 1 && action_y == 0) {
        return {1, -1};
    } else if (action_x == 1 && action_y == 2) {
        return {-1, 1};
    } else if (action_x == 2 && action_y == 0) {
        return {-1, 1};
    } else if (action_x == 2 && action_y == 1) {
        return {1, -1};
    } else {
        throw std::invalid_argument("Invalid action");
    }
}
