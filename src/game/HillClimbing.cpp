#include "HillClimbing.h"
#include "../agent/Monotone.h"
#include <random>
#include <tuple>

HillClimbing::HillClimbing() {
    std::tie(probability_matrix, reward_matrix1, reward_matrix2) = initialize_game_rewards();
}

std::tuple<std::array<std::array<double, 3>, 3>, std::array<std::array<double, 3>, 3>, std::array<std::array<double, 3>, 3>>
HillClimbing::initialize_game_rewards() {
    std::array<std::array<double, 3>, 3> probability_matrix{};
    std::array<std::array<double, 3>, 3> reward_matrix1{};
    std::array<std::array<double, 3>, 3> reward_matrix2{};

    probability_matrix[0] = {0.4, 0.25, 0.6};
    probability_matrix[1] = {0.25, 0.8, 0.8};
    probability_matrix[2] = {0.7, 0.6, 0.8};

    reward_matrix1[0] = {-3.5, -46, -6};
    reward_matrix1[1] = {-46, -5, -5};
    reward_matrix1[2] = {-4, -6, -6};

    reward_matrix2[0] = {4, -38, -16};
    reward_matrix2[1] = {-38, 5, 0};
    reward_matrix2[2] = {-17, -16, -1};

    return {probability_matrix, reward_matrix1, reward_matrix2};
}


Rewards HillClimbing::step(Action action_x, Action action_y) {
    double probability = probability_matrix[action_x][action_y];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    double reward = 0.0;

    if (probability >= dis(gen)) {
        reward = reward_matrix1[action_x][action_y];
    } else {
        reward = reward_matrix2[action_x][action_y];
    }

    return {reward, reward};
}