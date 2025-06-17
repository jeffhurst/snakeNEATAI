// NEAT.cpp
#include "NEAT.h"
#include <algorithm>
#include <cmath>
using namespace neat;

//static constexpr float C1=1.0f, C2=1.0f, C3=0.4f, COMPAT_THRESH=3.0f;

NEAT::NEAT(int popSize, int inN, int outN)
 : popSize_(popSize), rng_(std::random_device{}())
{
    // initialize genomes with minimal structure
    for (int i = 0; i < popSize_; ++i) {
        Genome* g = new Genome();
        // add input and output nodes
        for (NodeId j=0; j<inN; ++j) g->nodes[j] = {j, NodeGene::INPUT};
        for (NodeId j=0; j<outN; ++j) g->nodes[inN+j] = {inN+j, NodeGene::OUTPUT};
        population_.push_back(g);
    }
}

NEAT::~NEAT() {
    for (auto* g: population_) delete g;
}

void NEAT::epoch(std::function<void(Genome&)> evalFunc) {
    // evaluate
    for (auto* g : population_) {
        evalFunc(*g);
    }
    // sort by fitness descending
    std::sort(population_.begin(), population_.end(),
              [](Genome* a, Genome* b){ return a->fitness > b->fitness; });
    // speciate & reproduce
    speciate();
    reproduce();
    generation++;
}

Genome* NEAT::getBest() const {
    return population_.front();
}

void NEAT::speciate() {
    for (auto& s : species_) s.reset();
    species_.clear();
    for (auto* g : population_) {
        bool placed = false;
        for (auto& s : species_) {
            // compatibility = ??? (stub, always join first)
            placed = true;
            s.members.push_back(g);
            break;
        }
        if (!placed) {
            species_.push_back(Species());
            species_.back().members.push_back(g);
        }
    }
}

void NEAT::reproduce() {
    std::vector<Genome*> newPop;
    // elitism: keep top 2
    newPop.push_back(population_[0]);
    newPop.push_back(population_[1]);
    std::uniform_real_distribution<float> uni(0,1);
    while ((int)newPop.size() < popSize_) {
        // tournament select parents
        Genome* a = population_[std::uniform_int_distribution<int>(0,popSize_/2)(rng_)];
        Genome* b = population_[std::uniform_int_distribution<int>(0,popSize_/2)(rng_)];
        if (a->fitness < b->fitness) std::swap(a,b);
        Genome* child = new Genome(Genome::crossover(*a,*b));
        child->mutateWeights();
        if (uni(rng_) < 0.2f) child->mutateAddConnection();
        if (uni(rng_) < 0.01f) child->mutateAddNode();
        newPop.push_back(child);
    }
    // delete old non-elite
    for (size_t i=2; i<population_.size(); ++i) delete population_[i];
    population_.swap(newPop);
}
