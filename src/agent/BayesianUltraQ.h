//
// Created by ward on 1/3/23.
//

#ifndef HYPER_Q_BAYESIANULTRAQ_H
#define HYPER_Q_BAYESIANULTRAQ_H

#include "BayesianHyperQ.h"
#include "../utils/json.hpp"

using json = nlohmann::json;

enum class Similarity {
    likelihood,
    likelihood_scaled,
    posterior,
    cosine,
};

NLOHMANN_JSON_SERIALIZE_ENUM(Similarity, {
    { Similarity::likelihood, "likelihood" },
    { Similarity::likelihood_scaled, "likelihood_scaled" },
    { Similarity::posterior, "posterior" },
    { Similarity::cosine, "cosine" },
})

class BayesianUltraQ : public BayesianHyperQ {
public:
    BayesianUltraQ(double alpha, double gamma, double mu, Similarity similarity, std::optional<double> init)
            : BayesianHyperQ(alpha, gamma, mu, init), similarity(similarity) {}

    double observe(Reward r, Action action_x, Strategy original_x, Action action_y, Strategy _y) override {
        update_posterior(action_y);

        auto max = greedy().second;
        auto sum = 0.0;
        for (StrategyIndex x_index = 0; x_index < num_strategies; ++x_index) {
            if (!valid_index(x_index)) continue;
            auto x = index_to_strategy(x_index);

            for (StrategyIndex y_index = 0; y_index < num_strategies; ++y_index) {
                if (!valid_index(y_index)) continue;
                auto index = strategies_to_index({x_index, y_index});

                auto s = 0.0;
                switch (similarity) {
                    case Similarity::likelihood:
                        s = x[action_x];
                        break;
                    case Similarity::likelihood_scaled:
                        s = x[action_x] * original_x[action_x];
                        break;
                    case Similarity::posterior:
                        throw std::invalid_argument("Not implemented");
                    case Similarity::cosine:
                        for (unsigned int i = 0; i < x.size(); ++i)
                            s += x[i] * original_x[i];
                        break;
                };

                auto bellman_error = s * posterior_table[y_index] * (r + gamma * max - hyper_q_table[index]);
                hyper_q_table[index] += alpha * bellman_error;
                sum += bellman_error;
            }
        }

        return sum;
    }

private:
    Similarity similarity;
};


#endif //HYPER_Q_BAYESIANULTRAQ_H
