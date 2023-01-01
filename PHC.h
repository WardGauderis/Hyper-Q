#ifndef PHC_H
#define PHC_H

#include "Agent.h"
#include "Definitions.h"
#include <memory>

static const unsigned int num_states = 9;
static const unsigned int num_actions = 3;

class PHC : public Agent {
    public:
        PHC(float alpha_, float delta_, float gamma_, float epsilon_);

        std::pair<Action, Strategy> act() override;
        void observe(Reward r, Strategy x, Action action_y, Strategy y) override;

    private:
        // parameters.
        float alpha;
        float delta;
        float gamma;
        float epsilon;
        unsigned long current_state;
        unsigned long current_action;
        // tables.
        float q_table[num_states][num_actions];
        Strategy policy_table[num_states];

        // methods.
        unsigned long greedy();
};

#endif //PHC_H
