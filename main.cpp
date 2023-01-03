#include <iostream>

#include "Game.h"
#include "Agent.h"
#include "RockPaperScissors.h"
#include "HyperQ.h"
#include "BayesianHyperQ.h"
#include "EMA.h"
#include "Omniscient.h"
#include "Monotone"
#include "PHC.h"
#include "IGA.h"
#include "CooperationGame.h"
#include "BayesianUltraQ.h"

#include <memory>

//std::string ROOT = R"(./results analysis/results/)";
std::string ROOT = R"(C:\GitHub\Hyper-Q\results analysis\results2\)";

void run_test(const std::string &output_file,
              const unsigned int steps,
              const std::unique_ptr<Game> &game,
              const std::unique_ptr<Agent> &agent_x,
              const std::unique_ptr<Agent> &agent_y,
              const unsigned int log_every = 10
) {
    std::cout << "############################################################################" << std::endl;
    std::cout << "Running test: " << output_file << std::endl;
    std::cout << "############################################################################" << std::endl;

    std::ofstream output;
    output.open(output_file);

    for (unsigned int i = 0; i < steps; i++) {
        Action action_x, action_y;
        Strategy strategy_x, strategy_y;
        Reward value_x, value_y = 0.0;

        if (i % 1000 == 0) {
            std::tie(action_x, strategy_x, value_x) = agent_x->random_restart();
        } else {
            std::tie(action_x, strategy_x, value_x) = agent_x->act();
        }
        if (i % 1000 == 500) {
            std::tie(action_y, strategy_y, value_y) = agent_y->random_restart();
        } else {
            std::tie(action_y, strategy_y, value_y) = agent_y->act();
        }

        auto [reward_x, reward_y] = game->step(action_x, action_y);

        if (i % log_every == 0) {
            output << action_x << " " << action_y << " " << reward_x << " " << reward_y << " "
                   << strategy_x[0] << " " << strategy_x[1] << " " << strategy_x[2] << " "
                   << strategy_y[0] << " " << strategy_y[1] << " " << strategy_y[2] << " "
                   << value_x << " " << value_y << "\n";
        }

        agent_x->observe(reward_x, action_x, strategy_x, action_y, strategy_y);
        agent_y->observe(reward_y, action_y, strategy_y, action_x, strategy_x);

        if (i % 10000 == 1) {
            std::cout << "Step " << i << std::endl;
            std::cout << "Strategy x: " << strategy_x[0] << " " << strategy_x[1] << " " << strategy_x[2] << std::endl;
            std::cout << "Strategy y: " << strategy_y[0] << " " << strategy_y[1] << " " << strategy_y[2] << std::endl;
        }
    }
}

int extension() {
    std::unique_ptr<Game> game = std::make_unique<CooperationGame>();

    auto gamma = 0.9;
    auto alpha = 0.01;
    auto mu = 0.005;

    auto experiments = 20;
    auto steps = 600000;

    // EMA vs EMA: COOP
    for (int i = 0; i < experiments; i++) {
        std::cout << "Experiment " << i << std::endl;

        srand(static_cast<unsigned int>(i));
        std::unique_ptr<Agent> agent_x = std::make_unique<HyperQ>(std::make_unique<EMA>(mu), alpha, gamma);
        std::unique_ptr<Agent> agent_y = std::make_unique<HyperQ>(std::make_unique<EMA>(mu), alpha, gamma);


        std::stringstream output_file;
        output_file << ROOT
                    << R"(cooperation\EMA vs EMA\experiment_)"
                    << i
                    << ".txt";

        // Run the test and store the output in the output file
        run_test(output_file.str(), static_cast<unsigned int>(steps), game, agent_x, agent_y);
    }

    // Omniscient vs Omniscient: COOP
    for (int i = 0; i < experiments; i++) {
        srand(static_cast<unsigned int>(i));
        std::unique_ptr<Agent> agent_x = std::make_unique<HyperQ>(std::make_unique<Omniscient>(), alpha, gamma);
        std::unique_ptr<Agent> agent_y = std::make_unique<HyperQ>(std::make_unique<Omniscient>(), alpha, gamma);

        std::stringstream output_file;
        output_file << ROOT
                    << R"(cooperation\Omniscient vs Omniscient\experiment_)"
                    << i
                    << ".txt";

        // Run the test and store the output in the output file
        run_test(output_file.str(), static_cast<unsigned int>(steps), game, agent_x, agent_y);
    }

    // Bayesian vs Bayesian: COOP
    for (int i = 0; i < 3; i++) {
        srand(static_cast<unsigned int>(i));
        std::unique_ptr<Agent> agent_x = std::make_unique<BayesianHyperQ>(alpha, gamma, mu);
        std::unique_ptr<Agent> agent_y = std::make_unique<BayesianHyperQ>(alpha, gamma, mu);

        std::stringstream output_file;
        output_file << ROOT
                    << R"(cooperation\Bayesian vs Bayesian2\experiment_)"
                    << i
                    << ".txt";

        // Run the test and store the output in the output file
        run_test(output_file.str(), static_cast<unsigned int>(steps), game, agent_x, agent_y);
    }

    return 0;
}

int main() {

    std::unique_ptr<Game> game = std::make_unique<RockPaperScissors>();

    auto gamma = 0.99;
    auto alpha = 0.1;
    auto mu = 0.01;

    auto delta = 0.01;
    auto epsilon = 0.01;
    auto step_size = 0.01;

    auto experiments = 20;
    unsigned int steps = 600000;

    // Bayes vs monotone
    // Omniscient vs monotone
    // EMA vs monotone
    // PHC vs monotone
    // IGA vs monotone

    if (true) {
        // Bayesian Hyper-Q vs monotone
        for (int i = 0; i < 3; i++) {

            srand(static_cast<unsigned int>(i));
            std::unique_ptr<Agent> agent_x = std::make_unique<BayesianUltraQ>(alpha, gamma, mu);
            std::unique_ptr<Agent> agent_y = std::make_unique<Monotone>(Strategy{0, 1, 0});

            std::stringstream output_file;
            output_file << ROOT
                        << R"(Bayesian vs monotone/experiment_)" << i
                        << ".txt";

            // Run the test and store the output in the output file
            run_test(output_file.str(), steps, game, agent_x, agent_y);

        }
    }

    if (true) {
        // Omniscient vs monotone
        for (int i = 0; i < experiments; i++) {
            srand(static_cast<unsigned int>(i));
            std::unique_ptr<Agent> agent_x = std::make_unique<HyperQ>(std::make_unique<Omniscient>(), alpha, gamma);
            std::unique_ptr<Agent> agent_y = std::make_unique<Monotone>(Strategy{0, 0, 1});

            std::stringstream output_file;
            output_file << ROOT
                        << R"(Omniscient vs monotone/experiment_)" << i
                        << ".txt";

            // Run the test and store the output in the output file
            run_test(output_file.str(), steps, game, agent_x, agent_y);
        }
    }

    if (true) {
        // EMA vs monotone
        for (int i = 0; i < experiments; i++) {
            srand(static_cast<unsigned int>(i));
            std::unique_ptr<Agent> agent_x = std::make_unique<HyperQ>(std::make_unique<EMA>(mu), alpha, gamma);
            std::unique_ptr<Agent> agent_y = std::make_unique<Monotone>(Strategy{0, 0, 1});

            std::stringstream output_file;
            output_file << ROOT
                        << R"(EMA vs monotone/experiment_)" << i
                        << ".txt";

            // Run the test and store the output in the output file
            run_test(output_file.str(), steps, game, agent_x, agent_y);

        }
    }

    if (true) {
        // PHC vs monotone
        for (int i = 0; i < experiments; i++) {
            srand(static_cast<unsigned int>(i));
            std::unique_ptr<Agent> agent_x = std::make_unique<PHC>(alpha, delta, gamma, epsilon);
            std::unique_ptr<Agent> agent_y = std::make_unique<Monotone>(Strategy{0, 0, 1});


            std::stringstream output_file;
            output_file << ROOT
                        << R"(PHC vs monotone/experiment_)" << i
                        << ".txt";

            // Run the test and store the output in the output file
            run_test(output_file.str(), steps, game, agent_x, agent_y);

        }
    }

    if (true) {
        // IGA vs monotone
        for (int i = 0; i < experiments; i++) {
            srand(static_cast<unsigned int>(i));
            std::unique_ptr<Agent> agent_x = std::make_unique<IGA>(step_size);
            std::unique_ptr<Agent> agent_y = std::make_unique<Monotone>(Strategy{0, 1, 0});


            std::stringstream output_file;
            output_file << ROOT
                        << R"(IGA vs monotone/experiment_)" << i
                        << ".txt";

            // Run the test and store the output in the output file
            run_test(output_file.str(), steps, game, agent_x, agent_y);
        }
    }








    // PHC VS Bayes
    // PHC VS Omniscient
    // PHC VS EMA

    if (true) {
        // PHC vs Bayesian Hyper-Q
        for (int i = 0; i < 3; i++) {

            srand(static_cast<unsigned int>(time(nullptr)));
            std::unique_ptr<Agent> agent_x = std::make_unique<PHC>(alpha, delta, gamma, epsilon);
            std::unique_ptr<Agent> agent_y = std::make_unique<BayesianHyperQ>(alpha, gamma, mu);


            std::stringstream output_file;
            output_file << ROOT
                        << R"(PHC vs Bayesian/experiment_)" << i
                        << ".txt";

            // Run the test and store the output in the output file
            run_test(output_file.str(), steps, game, agent_x, agent_y);

        }
    }

    if (true) {
        // PHC vs EMA Hyper-Q
        for (int i = 0; i < 20; i++) {
            srand(static_cast<unsigned int>(i));
            std::unique_ptr<Agent> agent_x = std::make_unique<PHC>(alpha, delta, gamma, epsilon);
            std::unique_ptr<Agent> agent_y = std::make_unique<HyperQ>(std::make_unique<EMA>(mu), alpha, gamma);

            std::stringstream output_file;
            output_file << ROOT
                        << R"(PHC vs EMA/experiment_)" << i
                        << ".txt";

            // Run the test and store the output in the output file
            run_test(output_file.str(), steps, game, agent_x, agent_y);
        }
    }

    if (true) {
        // PHC vs Omniscient Hyper-Q
        for (int i = 0; i < 20; i++) {
            srand(static_cast<unsigned int>(i));
            std::unique_ptr<Agent> agent_x = std::make_unique<PHC>(alpha, delta, gamma, epsilon);
            std::unique_ptr<Agent> agent_y = std::make_unique<HyperQ>(std::make_unique<Omniscient>(), alpha, gamma);

            std::stringstream output_file;
            output_file << ROOT
                        << R"(PHC vs omniscient/experiment_)" << i
                        << ".txt";

            // Run the test and store the output in the output file
            run_test(output_file.str(), steps, game, agent_x, agent_y);
        }
    }









    // IGA VS Bayes
    // IGA VS Omniscient
    // IGA VS EMA

    if (true) {
        // IGA vs Bayesian Hyper-Q
        for (int i = 0; i < 3; i++) {

            srand(static_cast<unsigned int>(time(nullptr)));
            std::unique_ptr<Agent> agent_x = std::make_unique<IGA>(step_size);
            std::unique_ptr<Agent> agent_y = std::make_unique<BayesianHyperQ>(alpha, gamma, mu);


            std::stringstream output_file;
            output_file << ROOT
                        << R"(IGA vs Bayesian/experiment_)" << i
                        << ".txt";

            // Run the test and store the output in the output file
            run_test(output_file.str(), steps, game, agent_x, agent_y);

        }
    }

    if (true) {
        // IGA vs EMA Hyper-Q
        for (int i = 0; i < 20; i++) {

            srand(static_cast<unsigned int>(time(nullptr)));
            std::unique_ptr<Agent> agent_x = std::make_unique<IGA>(step_size);
            std::unique_ptr<Agent> agent_y = std::make_unique<HyperQ>(std::make_unique<EMA>(mu), alpha, gamma);


            std::stringstream output_file;
            output_file << ROOT
                        << R"(IGA vs EMA/experiment_)" << i
                        << ".txt";

            // Run the test and store the output in the output file
            run_test(output_file.str(), steps, game, agent_x, agent_y);

        }
    }

    if (true) {
        // IGA vs Omniscient Hyper-Q
        for (int i = 0; i < 20; i++) {

            srand(static_cast<unsigned int>(time(nullptr)));
            std::unique_ptr<Agent> agent_x = std::make_unique<IGA>(step_size);
            std::unique_ptr<Agent> agent_y = std::make_unique<HyperQ>(std::make_unique<Omniscient>(), alpha, gamma);

            std::stringstream output_file;
            output_file << ROOT
                        << R"(IGA vs omniscient/experiment_)" << i
                        << ".txt";

            // Run the test and store the output in the output file
            run_test(output_file.str(), steps, game, agent_x, agent_y);

        }
    }







    if (false)
        extension();


    return 0;
}