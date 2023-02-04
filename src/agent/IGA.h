#ifndef IGA_H
#define IGA_H

#include "Agent.h"
#include "../utils/Definitions.h"
#include <memory>

class IGA : public Agent {
    public:
        IGA(float step_size_);

        std::tuple<Action, Strategy, Reward> act() override;
        double observe(Reward r, Action action_x, Strategy x, Action action_y, Strategy y) override;

    private:
        // parameters.
        double step_size;
        // tables.
        Strategy policy;
};

#endif //IGA_H