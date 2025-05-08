#pragma once
#include "Genome.h"
#include "Network.h"
#include "Species.h"
#include <vector>
#include <random>

namespace neat {

struct NEAT {
    NEAT(int popSize, int inN, int outN);
    ~NEAT();
    void epoch(std::function<void(Genome&)> evalFunc);
    Genome* getBest() const;
    const std::vector<Species>& species() const { return species_; }
    int generation = 0;
private:
    int popSize_;
    std::vector<Genome*> population_;
    std::vector<Species> species_;
    std::mt19937 rng_;
    void speciate();
    void reproduce();
};

} // namespace neat
