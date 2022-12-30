#include <iostream>

#include "Game.h"
#include "Agent.h"
#include "RockPaperScissors.h"
#include "HyperQ.h"
#include "EMA.h"
#include "Bayesian.h"
#include "Omniscient.h"

#include <fstream>
#include <memory>

std::pair<Action, Strategy> random_action() {
    auto action = static_cast<Action>(rand() % 3);
    auto strategy = Strategy{1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
    return {action, strategy};
}

void run_test(const std::string &output_file,
              const unsigned int steps,
              const std::unique_ptr<Game> &game,
              const std::unique_ptr<Agent> &agent_x,
              const std::unique_ptr<Agent> &agent_y) {
    std::ofstream output;
    output.open(output_file);

    for (unsigned int i = 0; i < steps; i++) {
        Action action_x, action_y;
        Strategy strategy_x, strategy_y;
        if (i % 1000 == 0) {
            std::tie(action_x, strategy_x) = random_action();
            std::tie(action_y, strategy_y) = random_action();
        } else {
            std::tie(action_x, strategy_x) = agent_x->act();
            std::tie(action_y, strategy_y) = agent_y->act();
        }

        auto [reward_x, reward_y] = game->step(action_x, action_y);

        output << action_x << " " << action_y << " " << reward_x << " " << reward_y << "\n";

        agent_x->observe(reward_x, strategy_x, action_y, strategy_y);
        agent_y->observe(reward_y, strategy_y, action_x, strategy_x);
    }
}

int main() {
    std::unique_ptr<Game> game = std::make_unique<RockPaperScissors>();

    auto gamma = 0.9;
    auto alpha = 0.01;
    auto mu = 0.005;

//    std::unique_ptr<Agent> agent_x = std::make_unique<HyperQ>(std::make_unique<Omniscient>(), alpha, gamma);
//    std::unique_ptr<Agent> agent_y = std::make_unique<HyperQ>(std::make_unique<Omniscient>(), alpha, gamma);
    std::unique_ptr<Agent> agent_x = std::make_unique<HyperQ>(std::make_unique<EMA>(mu), alpha, gamma);
    std::unique_ptr<Agent> agent_y = std::make_unique<HyperQ>(std::make_unique<EMA>(mu), alpha, gamma);

    run_test("output.txt", 1000000, game, agent_x, agent_y);
    return 0;
}
