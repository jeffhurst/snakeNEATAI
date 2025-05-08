#pragma once
#include "Snake.h"
#include <vector>
#include <random>

namespace game {

struct EvalResult {
    double fitness;
    std::vector<Snake::Vec2i> bestPath; // for visualization
};

class Game {
public:
    Game(int gridW, int gridH, int maxTicks);
    // Run one simulation for given neural network; return fitness & path
    template<typename NetworkT>
    EvalResult evaluate(NetworkT& net);
private:
    int gridW_, gridH_, maxTicks_;
    std::mt19937 rng_;
};
} // namespace game
