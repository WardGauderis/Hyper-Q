#ifndef IGA_H
#define IGA_H

#include "Agent.h"
#include "Definitions.h"
#include <memory>

class IGA : public Agent {
    public:
        IGA(float step_size_);

        std::tuple<Action, Strategy, Reward> act() override;
        void observe(Reward r, Strategy x, Action action_y, Strategy y) override;

    private:
        // parameters.
        float step_size;
        // tables.
        Strategy policy;

        // methods
        int greedy(Strategy strat);
};

#endif //IGA_H