#include <iostream>

#include "Game.h"
#include "Agent.h"
#include "RockPaperScissors.h"
#include "HyperQ.h"
#include "EMA.h"
#include "Bayesian.h"
#include "Omniscient.h"
#include "Monotone"
#include "CooperationGame.h"

#include <fstream>
#include <memory>
#include <sstream>

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
            std::tie(action_x, strategy_x) = agent_x->random_restart();
            std::tie(action_y, strategy_y) = agent_y->random_restart();
        } else {
            std::tie(action_x, strategy_x) = agent_x->act();
            std::tie(action_y, strategy_y) = agent_y->act();
        }

        auto [reward_x, reward_y] = game->step(action_x, action_y);

        output << action_x << " " << action_y << " " << reward_x << " " << reward_y << "\n";

        agent_x->observe(reward_x, strategy_x, action_y, strategy_y);
        agent_y->observe(reward_y, strategy_y, action_x, strategy_x);

        if (i % 100000 == 0) {
            std::cout << "Step " << i << std::endl;
        }
    }
}

int extension() {
    std::unique_ptr<Game> game = std::make_unique<CooperationGame>();

    auto gamma = 0.9;
    auto alpha = 0.01;
    auto mu = 0.005;

    auto experiments = 1;

    // EMA vs EMA: COOP
    for (int i = 0; i < experiments; i++) {
        srand(i);
        std::unique_ptr<Agent> agent_x = std::make_unique<HyperQ>(std::make_unique<EMA>(mu), alpha, gamma);
        std::unique_ptr<Agent> agent_y = std::make_unique<HyperQ>(std::make_unique<EMA>(mu), alpha, gamma);


        std::stringstream output_file;
        output_file << R"(C:\GitHub\Hyper-Q\results analysis\results\cooperation\EMA vs EMA\experiment_)" << i
                    << ".txt";

        // Run the test and store the output in the output file
        run_test(output_file.str(), 1500000, game, agent_x, agent_y);
    }

    // Omniscient vs Omniscient: COOP
    for (int i = 0; i < experiments; i++) {
        srand(i);
        std::unique_ptr<Agent> agent_x = std::make_unique<HyperQ>(std::make_unique<Omniscient>(), alpha, gamma);
        std::unique_ptr<Agent> agent_y = std::make_unique<HyperQ>(std::make_unique<Omniscient>(), alpha, gamma);

        std::stringstream output_file;
        output_file << R"(C:\GitHub\Hyper-Q\results analysis\results\cooperation\Omniscient vs Omniscient\experiment_)" << i
                    << ".txt";

        // Run the test and store the output in the output file
        run_test(output_file.str(), 1500000, game, agent_x, agent_y);
    }

}


int main() {

    std::unique_ptr<Game> game = std::make_unique<RockPaperScissors>();

    auto gamma = 0.9;
    auto alpha = 0.01;
    auto mu = 0.005;

    auto experiments = 20;

    if (false) {
        // Omniscient vs monotone
        for (int i = 0; i < experiments; i++) {
            srand(i);
            std::unique_ptr<Agent> agent_x = std::make_unique<HyperQ>(std::make_unique<Omniscient>(), alpha, gamma);
            std::unique_ptr<Agent> agent_y = std::make_unique<Monotone>(Strategy{0, 0, 1});

            std::stringstream output_file;
            output_file << R"(Omniscient vs monotone\experiment_)" << i
                        << ".txt";

            // Run the test and store the output in the output file
            run_test(output_file.str(), 1500000, game, agent_x, agent_y);
        }
    }

    if (false) {
        // EMA vs monotone
        for (int i = 0; i < experiments; i++) {
            srand(i);
            std::unique_ptr<Agent> agent_x = std::make_unique<HyperQ>(std::make_unique<EMA>(mu), alpha, gamma);
            std::unique_ptr<Agent> agent_y = std::make_unique<Monotone>(Strategy{0, 0, 1});

            std::stringstream output_file;
            output_file << R"(EMA vs monotone\experiment_)" << i
                        << ".txt";

            // Run the test and store the output in the output file
            run_test(output_file.str(), 1500000, game, agent_x, agent_y);
        }
    }


    if (true)
        extension();


    return 0;
}
