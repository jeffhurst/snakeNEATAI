#include "Game.h"
#include "Snake.h"
#include <cmath>
#include <iostream>
#include <random> 

using namespace game;

Game::Game(int w, int h, int maxT)
 : gridW_(w), gridH_(h), maxTicks_(maxT)
{

}

template<typename N>
EvalResult Game::evaluate(N& net) {
    // each thread gets its own RNG, seeded once
    static thread_local std::mt19937 local_rng(std::random_device{}());

    Snake snake(gridW_, gridH_);
    std::uniform_int_distribution<int> distX(0, gridW_-1),
                                     distY(0, gridH_-1);
    Vec2i food{distX(local_rng), distY(local_rng)};
    double fitness = 0;
    std::vector<Vec2i> path;
    for (int t = 0; t < maxTicks_; ++t) {
        // prepare inputs: normalized head pos, food delta
        float hx = float(snake.head().x)/gridW_,
              hy = float(snake.head().y)/gridH_,
              fx = float(food.x - snake.head().x)/gridW_,
              fy = float(food.y - snake.head().y)/gridH_;
        auto outputs = net.feed({hx, hy, fx, fy});
        // pick largest output -> direction
        int dir = std::distance(outputs.begin(),
            std::max_element(outputs.begin(), outputs.end()));
        snake.setDirection(static_cast<Dir>(dir));
        if (!snake.update()) break;  // died
        // ate food?
        if (snake.head().x == food.x && snake.head().y == food.y) {
            snake.grow();
            fitness += 100.0;
            food = {distX(local_rng), distY(local_rng)};
        }
        // incremental fitness: survival + closeness to food
        double dist = std::hypot(food.x - snake.head().x, food.y - snake.head().y);
        fitness += 1.0 - (dist / std::hypot(gridW_, gridH_));
        if (t % 10 == 0)
            path.push_back(snake.head());
    }
    return {fitness, path};
}

#include "neat/Network.h"

namespace game {
  // force MSVC to emit the evaluate<Network> symbol
  template EvalResult Game::evaluate<neat::Network>(neat::Network& net);
}

// Explicit instantiation for our Network type will go in main.cpp.
