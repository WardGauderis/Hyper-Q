#include <iostream>

#include "game/Game.h"
#include "agent/Agent.h"
#include "agent/BayesianUltraQ.h"
#include "exploration/ExplorationMethod.h"

#include <memory>

#include "utils/json.hpp"

using json = nlohmann::json;

void run_test(const std::string &output_file,
              const unsigned int steps,
              const std::unique_ptr<Game> &game,
              const std::unique_ptr<ExplorationMethod> &exploration,
              const std::unique_ptr<Agent> &agent_x,
              const std::unique_ptr<Agent> &agent_y,
              const unsigned int log_every = 10
) {
//    std::cout << "############################################################################" << std::endl;
    std::cout << "Running test: " << output_file << std::endl;
//    std::cout << "############################################################################" << std::endl;

    std::ofstream output;
    output.open(output_file);

    double average_reward_x = 0.0, average_reward_y = 0.0;
    double average_bellman_error_x = 0.0, average_bellman_error_y = 0.0;

    for (unsigned int i = 0; i < steps; i++) {
        auto r = exploration->act(agent_x, agent_y, i);

        auto [reward_x, reward_y] = game->step(r.action_x, r.action_y);

        average_reward_x += (reward_x - average_reward_x) / (i + 1);
        average_reward_y += (reward_y - average_reward_y) / (i + 1);

        auto bellman_error_x = agent_x->observe(reward_x, r.action_x, r.strategy_x, r.action_y, r.strategy_y);
        auto bellman_error_y = agent_y->observe(reward_y, r.action_y, r.strategy_y, r.action_x, r.strategy_x);

//        average_bellman_error_x += (bellman_error_x - average_bellman_error_x) / (i + 1);
//        average_bellman_error_y += (bellman_error_y - average_bellman_error_y) / (i + 1);

        average_bellman_error_x = 0.05 * bellman_error_x + 0.95 * average_bellman_error_x;
        average_bellman_error_y = 0.05 * bellman_error_y + 0.95 * average_bellman_error_y;

        if (i % log_every == log_every - 1) {
            output << r.action_x << " " << r.action_y << " " << reward_x << " " << reward_y << " "
                   << r.strategy_x[0] << " " << r.strategy_x[1] << " " << r.strategy_x[2] << " "
                   << r.strategy_y[0] << " " << r.strategy_y[1] << " " << r.strategy_y[2] << " "
                   << bellman_error_x << " " << bellman_error_y << "\n";
        }

        if (false) {
//            if (i % 500000 == 500000 - 1) {
            std::cout << "Step " << i << ", output: " << output_file << std::endl;
            std::cout << "Strategy x: " << r.strategy_x[0] << " " << r.strategy_x[1] << " " << r.strategy_x[2]
                      << " Bellman: "
                      << average_bellman_error_x << " Reward: " << average_reward_x << std::endl;
            std::cout << "Strategy y: " << r.strategy_y[0] << " " << r.strategy_y[1] << " " << r.strategy_y[2]
                      << " Bellman: "
                      << average_bellman_error_y << " Reward: " << average_reward_y << std::endl;
        }
    }
}

int main() {
    std::ifstream f("config.json");
    const json j = json::parse(f);

    const unsigned int runs = j.at("runs");
    const unsigned int iterations = j.at("iterations");

#pragma omp parallel default(none) shared(j, runs, iterations)
    {
#pragma omp for
        for (unsigned int i = 0; i < runs; i++) {
            std::unique_ptr<Game> game = Game::from_json(j.at("game"));
            std::unique_ptr<ExplorationMethod> exploration = ExplorationMethod::from_json(j.at("exploration"));

            std::unique_ptr<Agent> agent_x = Agent::from_json(j.at("agent_x"));
            std::unique_ptr<Agent> agent_y = Agent::from_json(j.at("agent_y"));

            std::string output_file = "run_" + std::to_string(i) + ".txt";

            run_test(output_file, iterations, game, exploration, agent_x, agent_y);
        }
    }
    return 0;
}