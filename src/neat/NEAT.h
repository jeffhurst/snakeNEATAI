// NEAT.h
#pragma once
#include "Genome.h"
#include "Network.h"
#include "Species.h"
#include <vector>
#include <random>
#include <functional>

namespace neat {

struct NEAT {
    NEAT(int popSize, int inN, int outN);
    ~NEAT();

    // Evaluate+sort externally, then:
    void epoch(std::function<void(Genome&)> evalFunc);

    Genome* getBest() const;

    const std::vector<Species>& species()    const { return species_; }
    const std::vector<Genome*>& population() const { return population_; }
    int generation = 0;

private:
    int popSize_;
    std::vector<Genome*> population_;
    std::vector<Genome*> top10_;
    std::vector<Species> species_;
    std::mt19937 rng_;

    // speciation & reproduction params:
    float compatThreshold_;
    int   targetSpeciesCount_;
    int   stagnationLimit_;
    float thresholdAdjustStep_;

    // core steps:
    void speciate();
    void reproduce();

    // compute compatibility distance between two genomes
    float compatibilityDistance(const Genome& a, const Genome& b) const;
};

} // namespace neat
