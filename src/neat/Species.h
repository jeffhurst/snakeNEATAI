// Species.h
#pragma once
#include "Genome.h"
#include <vector>
#include <limits>

namespace neat {

struct Species {
    Genome*   representative = nullptr;   // chosen at start of each generation
    std::vector<Genome*> members;
    
    // stagnation tracking:
    float bestFitnessEver = -std::numeric_limits<float>::infinity();
    int   gensSinceImprovement = 0;

    // fitness sharing accumulator:
    double adjustedFitnessSum = 0.0;

    // call at start of speciation pass
    void resetForNextGen() {
        members.clear();
        adjustedFitnessSum = 0.0;
    }
};

} // namespace neat
