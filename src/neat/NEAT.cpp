// NEAT.cpp
#include "NEAT.h"
#include "InnovationTracker.h"
#include "NeatConfig.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
using namespace neat;

static constexpr float C1=1.0f, C2=1.0f, C3=0.4f, COMPAT_THRESH=3.0f;

NEAT::NEAT(int popSize, int inN, int outN)
 : popSize_(popSize), rng_(std::random_device{}())
{
    // --- 1) Decide on node ID ranges ---
    // Inputs:    [0 .. inN-1]
    // Bias:      [inN]
    // Outputs:   [inN+1 .. inN+outN]
    NodeId biasId        = inN;
    NodeId firstOutputId = inN + 1;
    NodeId nextFreeId    = inN + 1 + outN;

    // Tell the tracker not to hand out any node IDs < nextFreeId
    InnovationTracker::getInstance().initializeNodeCounter(nextFreeId);

    // Uniform random weight initializer in [-1, +1]
    std::uniform_real_distribution<float> weightDist(-1.0f, 1.0f);

    // --- 2) Create initial population ---
    for (int i = 0; i < popSize_; ++i) {
        Genome* g = new Genome();

        // 2a) Add all input nodes
        for (NodeId nid = 0; nid < inN; ++nid) {
            g->nodes[nid] = { nid, NodeGene::INPUT };
        }
        // 2b) Add the bias node
        g->nodes[biasId] = { biasId, NodeGene::BIAS };

        // 2c) Add all output nodes
        for (NodeId j = 0; j < outN; ++j) {
            NodeId outId = firstOutputId + j;
            g->nodes[outId] = { outId, NodeGene::OUTPUT };
        }

        // 2d) Fully connect each input + bias → every output
        for (NodeId src = 0; src <= inN; ++src) {            // 0..inN = inputs + bias
            for (NodeId dst = firstOutputId; 
                 dst < firstOutputId + outN; 
                 ++dst) 
            {
                // Ask the global tracker for a unique innovation number
                InnovId innov = InnovationTracker::getInstance()
                                     .getConnectionInnov(src, dst);

                // Create the connection with a random initial weight
                float w = weightDist(rng_);
                g->connections[innov] = { innov, src, dst, w, true };
            }
        }

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
        if (uni(rng_) < PROB_ADD_CONNECTION) {
            child->mutateAddConnection();
        }
        if (uni(rng_) < PROB_ADD_NODE){
               child->mutateAddNode();
        }
        newPop.push_back(child);
    }
    // delete old non-elite
    for (size_t i=2; i<population_.size(); ++i) delete population_[i];
    population_.swap(newPop);
}
