cmake_minimum_required(VERSION 3.24)
project(Hyper_Q)

set(CMAKE_CXX_STANDARD 17)

# O0: debug
# O3: optimized
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wunused -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2")

add_executable(Hyper_Q main.cpp Agent.h Game.h RockPaperScissors.h RockPaperScissors.cpp HyperQ.h HyperQ.cpp StrategyEstimation.h EMA.h Bayesian.h Omniscient.h Definitions.h PHC.h PHC.cpp IGA.h IGA.cpp Monotone)
