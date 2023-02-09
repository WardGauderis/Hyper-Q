cmake_minimum_required(VERSION 3.24)
project(Hyper_Q)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wunused -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2")

add_executable(Hyper_Q src/main.cpp src/agent/Agent.h src/game/Game.h src/game/RockPaperScissors.h src/game/RockPaperScissors.cpp src/agent/HyperQ.h src/estimation/StrategyEstimation.h src/estimation/EMA.h src/estimation/Omniscient.h src/utils/Definitions.h src/agent/Monotone.h src/agent/BayesianHyperQ.h src/agent/Agent.cpp src/agent/PHC.h src/agent/PHC.cpp src/agent/IGA.h src/agent/IGA.cpp src/game/HillClimbing.cpp src/agent/BayesianUltraQ.h src/utils/json.hpp src/game/Game.cpp src/exploration/ExplorationMethod.h src/exploration/ExplorationMethod.cpp src/estimation/StrategyEstimation.cpp)
