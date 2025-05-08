// main.cpp
#include "core/ThreadPool.h"
#include "game/Game.h"
#include "neat/NEAT.h"
#include "neat/Network.h"
#include "render/Renderer.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv){
    const int GRID_W    = 20;
    const int GRID_H    = 20;
    const int MAX_TICKS = 200;
    const int POP       = 150;
    const int GENS      = 1000;
    const int INPUT_N   = 4;
    const int OUTPUT_N  = 4;
    const int THREADS   = std::thread::hardware_concurrency();

    neat::NEAT neat(POP, INPUT_N, OUTPUT_N);
    core::ThreadPool pool(THREADS);
    game::Game   simulator(GRID_W, GRID_H, MAX_TICKS);
    render::Renderer renderer(600, 600, GRID_W, GRID_H);

    bool paused      = false;
    float speed      = 0.1f;
    int   observeIdx = 0;

    std::mutex print_mtx;

    for (int gen = 0; gen < GENS && !renderer.shouldClose(); ++gen) {
        {
            std::lock_guard<std::mutex> lock(print_mtx);
            std::cout << "[LOG] Starting generation " << gen << "\n";
        }

        // reset counter
        std::atomic<int> completed{0};

        // 1) enqueue all POP evaluations
        for (int i = 0; i < POP; ++i) {
            pool.enqueue([&, i]() {
                auto* g = neat.population()[i];
                neat::Network net(*g);
                auto    res = simulator.evaluate(net);
                g->fitness   = res.fitness;

                // only print every 10th so we don't drown stdout
                if ((i % 10) == 0) {
                    std::lock_guard<std::mutex> lock(print_mtx);
                    std::cout << "[EVAL] gen=" << gen
                              << "  idx=" << i
                              << "  fitness=" << res.fitness
                              << "\n";
                }
                completed.fetch_add(1, std::memory_order_relaxed);
            });
        }

        // 2) wait — but still pump UI so window never hangs
        while (completed.load(std::memory_order_relaxed) < POP) {
            // draw a simple “waiting…” frame
            renderer.beginFrame();
            renderer.processUI(paused, speed, observeIdx);
            renderer.drawGrid();
            // you could draw a progress bar here… for now just stats text
            renderer.drawStats(gen,
                               0.0f,
                               /*avg*/ float(completed.load())/POP,
                               neat.species().size());
            renderer.endFrame();

            // small sleep to avoid burning 100% CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            if (renderer.shouldClose()) break;
        }

        // 3) now that every genome->fitness is set, call epoch once
        neat.epoch([](neat::Genome&){ /* no-op */ });

        {
            std::lock_guard<std::mutex> lock(print_mtx);
            std::cout << "[GEN] " << gen
                      << " – evaluation done. Best="
                      << neat.getBest()->fitness
                      << "\n[LOG]  Finished epoch; best fitness = "
                      << neat.getBest()->fitness
                      << "\n";
        }

        // render best one last time before moving on
        auto* best    = neat.getBest();
        neat::Network bn(*best);
        auto    br    = simulator.evaluate(bn);

        while (!renderer.shouldClose()) {
            renderer.beginFrame();
            renderer.processUI(paused, speed, observeIdx);
            renderer.drawGrid();
            renderer.drawSnake(br.bestPath);
            renderer.drawStats(gen,
                               best->fitness,
                               /*avg*/0,
                               neat.species().size());
            renderer.endFrame();
            if (!paused) break;
        }

        if (gen % 50 == 0) {
            std::lock_guard<std::mutex> lock(print_mtx);
            std::cout << "[LOG]  Exporting best_genome_" << gen << ".json\n";
            std::ofstream fout("best_genome_" + std::to_string(gen) + ".json");
            fout << "{ \"fitness\": " << best->fitness << " }";
        }
    }

    return 0;
}
