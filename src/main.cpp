// main.cpp

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <raylib.h>

// Project headers
#include "game/Game.h"
#include "game/Snake.h"
#include "neat/NEAT.h"
#include "neat/Network.h"
#include "render/Renderer.h"

int main() {
    // ------------------------------------------------------------------------
    // Simulation parameters (debug-friendly, small)
    // ------------------------------------------------------------------------
    const int GRID_W       = 10;    ///< grid width  (cells)
    const int GRID_H       = 10;    ///< grid height (cells)
    const int MAX_TICKS    = 100;   ///< max steps per simulation

    const int POP_SIZE     = 10000;    ///< genomes per generation
    const int INPUT_N      = 4;     ///< network input size (hx, hy, fx, fy)
    const int OUTPUT_N     = 4;     ///< network outputs (UP,DOWN,LEFT,RIGHT)
    const int GENERATIONS  = 200;   ///< total training generations

    // ------------------------------------------------------------------------
    // Rendering parameters
    // ------------------------------------------------------------------------
    const int SCREEN_W = 1200;       ///< game width  (pixels)
    const int SCREEN_H = 600;       ///< game height (pixels)

    // ------------------------------------------------------------------------
    // Initialize core systems
    // ------------------------------------------------------------------------
    game::Game     game(GRID_W, GRID_H, MAX_TICKS);
    neat::NEAT     neat(POP_SIZE, INPUT_N, OUTPUT_N);
    render::Renderer renderer(SCREEN_W, SCREEN_H, GRID_W, GRID_H);

    // ------------------------------------------------------------------------
    // Main generational loop
    // ------------------------------------------------------------------------
    while (!renderer.shouldClose() && neat.generation < GENERATIONS) {
        int gen = neat.generation;
        //std::cout << "\n===== Generation " << gen << " =====\n";

        double totalFitness = 0.0;
        double maxFitness   = -1e9;
        int    bestIdx      = 0;

        // Evaluate every genome
        auto pop = neat.population();
        for (size_t i = 0; i < pop.size(); ++i) {
            neat::Genome* g = pop[i];

            // Build network from genome
            neat::Network net(*g);

            // Run simulation and get fitness + sampled path
            game::EvalResult res = game.evaluate(net);
            g->fitness = res.fitness;
            totalFitness += res.fitness;

            // Track the best genome index
            if (res.fitness > maxFitness) {
                maxFitness = res.fitness;
                bestIdx    = static_cast<int>(i);
            }

            // // Verbose per-genome logging
            // std::cout << "[Gen " << gen << "] Genome " << i
            //           << " -> Fitness: " << res.fitness << "\n";
        }

        // Compute summary stats
        double avgFitness   = totalFitness / pop.size();
        int    speciesCount = static_cast<int>(neat.species().size());

        // // Summary log
        // std::cout << "Summaries for Generation " << gen << ":\n"
        //           << "  Max Fitness   = " << maxFitness   << "\n"
        //           << "  Avg Fitness   = " << avgFitness   << "\n"
        //           << "  Species Count = " << speciesCount << "\n";

        // --------------------------------------------------------------------
        // Visualization: re-evaluate best genome for path, then render a
        // single frame showing its sampled path + neural network + stats.
        // --------------------------------------------------------------------
        {
            neat::Genome* bestG = pop[bestIdx];
            neat::Network bestNet(*bestG);
            game::EvalResult bestRes = game.evaluate(bestNet);

            renderer.beginFrame();
            renderer.drawGrid();

            // Draw sampled head-positions (every 10 ticks) as a “snake”
            renderer.drawSnake(bestRes.bestPath);


            renderer.drawNetwork(bestNet);

            // Overlay generation stats on screen
            renderer.drawStats(
                gen,
                static_cast<float>(maxFitness),
                static_cast<float>(avgFitness),
                speciesCount
            );
            renderer.endFrame();
        }

        // --------------------------------------------------------------------
        // Speciate & reproduce to form the next generation
        // We pass a no-op evalFunc since fitness is already filled.
        // --------------------------------------------------------------------
        neat.epoch([](neat::Genome&){ /* already evaluated */ });
    }

    // ------------------------------------------------------------------------
    // FINAL DEMO: Best network plays Snake indefinitely until ESC
    // ------------------------------------------------------------------------
    {
        std::cout << "\n=== Training complete. Running final demonstration ===\n";
        neat::Genome* champion = neat.getBest();
        neat::Network  net(*champion);

        // Prepare demonstration environment
        game::Snake snake(GRID_W, GRID_H);
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> distX(0, GRID_W - 1),
                                          distY(0, GRID_H - 1);
        game::Vec2i food{ distX(rng), distY(rng) };

        // Lambda to reset snake & respawn food
        auto resetSim = [&]() {
            snake.reset();
            food = { distX(rng), distY(rng) };
        };

        resetSim();
        SetTargetFPS(5);


        // Demo loop: restart on death, exit on ESC
        while (!renderer.shouldClose()) {
            // 1) Get normalized inputs
            auto head = snake.head();
            float hx = float(head.x) / GRID_W;
            float hy = float(head.y) / GRID_H;
            float fx = float(food.x - head.x) / GRID_W;
            float fy = float(food.y - head.y) / GRID_H;

            // 2) Feed network and update direction
            auto outputs = net.feed({ hx, hy, fx, fy });
            int dir = std::distance(
                outputs.begin(),
                std::max_element(outputs.begin(), outputs.end())
            );
            snake.setDirection(static_cast<game::Dir>(dir));

            // 3) Advance snake; if collision, reset
            if (!snake.update()) {
                resetSim();
            } else {
                // Check for food
                if (snake.head().x == food.x && snake.head().y == food.y) {
                    snake.grow();
                    food = { distX(rng), distY(rng) };
                }
            }

            // 4) Render game + network
            renderer.beginFrame();
            renderer.drawGrid();
            renderer.drawSnake(snake.body());
            renderer.drawFood(food.x, food.y);
            renderer.drawNetwork(net);  // network rendering next to grid
            renderer.endFrame();

            // 5) Exit on ESC
            if (IsKeyPressed(KEY_ESCAPE)) {
                break;
            }
        }
    }

    return 0;
}
