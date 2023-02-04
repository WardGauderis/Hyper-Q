//
// Created by ward on 2/4/23.
//
#include "StrategyEstimation.h"
#include "Omniscient.h"
#include "EMA.h"

std::unique_ptr<StrategyEstimation> StrategyEstimation::from_json(const json &j) {
    std::string type = j.at("type");
    if (type == "omniscient") {
        return std::make_unique<Omniscient>();
    } else if (type == "ema") {
        double mu = j.at("mu");
        return std::make_unique<EMA>(mu);
    } else {
        throw std::invalid_argument("Unknown strategy estimation type: " + type);
    }
}

