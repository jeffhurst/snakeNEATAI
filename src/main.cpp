#include "core/ThreadPool.h"
#include "game/Game.h"
#include "neat/NEAT.h"
#include "neat/Network.h"
#include "render/Renderer.h"
#include <atomic>
#include <fstream>

int main(int argc, char** argv){
    const int GRID_W = 20, GRID_H = 20;
    const int MAX_TICKS = 200;
    const int POP = 150, GENS = 1000;
    const int INPUT_N = 4, OUTPUT_N = 4;
    const int THREADS = std::thread::hardware_concurrency();

    neat::NEAT neat(POP, INPUT_N, OUTPUT_N);
    core::ThreadPool pool(THREADS);
    game::Game simulator(GRID_W, GRID_H, MAX_TICKS);
    render::Renderer renderer(600, 600, GRID_W, GRID_H);

    bool paused = false;
    float speed = 1.0f;
    int observeIdx = 0;

    for (int gen = 0; gen < GENS && !renderer.shouldClose(); ++gen) {
        std::atomic<int> completed{0};
        std::vector<double> fitnesses(POP);
        std::vector<std::vector<game::Snake::Vec2i>> paths(POP);

        // evaluate in parallel
        neat.epoch([&](neat::Genome& g){
            neat::Network net(g);
            auto res = simulator.evaluate(net);
            g.fitness = res.fitness;
        });

        // after evaluation, render best
        neat::Genome* best = neat.getBest();
        neat::Network bestNet(*best);
        auto bestRes = simulator.evaluate(bestNet);

        // visualize generation
        while (!renderer.shouldClose()) {
            renderer.beginFrame();
            renderer.processUI(paused, speed, observeIdx);
            renderer.drawGrid();
            renderer.drawSnake(bestRes.bestPath);
            renderer.drawStats(gen, best->fitness, /*avg*/0, neat.species().size());
            renderer.endFrame();
            if (!paused) break;
        }

        // export top genome
        if (gen % 50 == 0) {
            std::ofstream fout("best_genome_" + std::to_string(gen) + ".json");
            // stub JSON serialization
            fout << "{ \"fitness\": " << best->fitness << " }";
        }
    }
    return 0;
}
