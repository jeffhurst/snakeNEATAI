// NEAT.cpp
#include "NEAT.h"
#include "InnovationTracker.h"
#include "NeatConfig.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <set>
using namespace neat;

// NEAT‐tuning constants
static constexpr float C1 = 1.0f;
static constexpr float C2 = 1.0f;
static constexpr float C3 = 0.4f;

// Initial compatibility threshold
static constexpr float INIT_COMPAT_THRESH   = 3.0f;
// Target number of species (for dynamic thresholding)
static constexpr int   TARGET_SPECIES_COUNT = 10;
// How many gens without improvement before we kill a species
static constexpr int   STAGNATION_LIMIT      = 100;
// How much to bump threshold each adjust step
static constexpr float THRESHOLD_STEP       = 0.3f;

NEAT::NEAT(int popSize, int inN, int outN)
 : popSize_(popSize),
   rng_(std::random_device{}()),
   compatThreshold_(INIT_COMPAT_THRESH),
   targetSpeciesCount_(TARGET_SPECIES_COUNT),
   stagnationLimit_(STAGNATION_LIMIT),
   thresholdAdjustStep_(THRESHOLD_STEP)
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
    // Now do an initial speciation so generation-0 species exist:
    speciate();
}

NEAT::~NEAT() {
    for (auto* g: population_) delete g;
}

void NEAT::epoch(std::function<void(Genome&)> evalFunc) {
    // 1) evaluate
    for (auto* g : population_) evalFunc(*g);

    // 2) sort by raw fitness descending
    std::sort(population_.begin(), population_.end(),
              [](Genome* a, Genome* b){ return a->fitness > b->fitness; });
    // 4) reproduce into next generation
    reproduce();
    // 2) immediately clear out every species’ member list
    for (auto& s : species_) {
        s.members.clear();
    }    
    // 3) speciate & prune stagnant species
    speciate();
    generation++;
}

Genome* NEAT::getBest() const {
    if (population_.size() < 2) {
        std::cerr << "ERROR: population_ has size " << population_.size() << " at generation " << generation << std::endl;
        exit(1);
        // Optionally: exit(1);
    }
    return population_.front();
}


float NEAT::compatibilityDistance(const Genome& A, const Genome& B) const {
    // gather all innovation IDs
    std::set<InnovId> allInnov;
    for (auto& kv : A.connections) allInnov.insert(kv.first);
    for (auto& kv : B.connections) allInnov.insert(kv.first);

    // find max innov in each
    InnovId maxA = A.connections.empty() ? 0 : A.connections.rbegin()->first;
    InnovId maxB = B.connections.empty() ? 0 : B.connections.rbegin()->first;

    int E = 0, D = 0;
    double Wdiff = 0;
    int matching = 0;

    for (InnovId innov : allInnov) {
        auto itA = A.connections.find(innov);
        auto itB = B.connections.find(innov);
        if (itA != A.connections.end() && itB != B.connections.end()) {
            // matching gene
            matching++;
            Wdiff += std::fabs(itA->second.weight - itB->second.weight);
        } else {
            // disjoint vs excess
            if (innov > maxA || innov > maxB) E++;
            else                               D++;
        }
    }

    double Wbar = matching>0 ? Wdiff / matching : 0.0;
    double N = std::max(A.connections.size(), B.connections.size());
    if (N < 20) N = 1;  // small‐genome normalization
    return (C1*E + C2*D) / N + C3 * Wbar;
}

void NEAT::speciate() {
    // adjust threshold to keep species count near target
    if (generation > 0) {
        if ((int)species_.size() > targetSpeciesCount_) 
            compatThreshold_ += thresholdAdjustStep_;
        else if ((int)species_.size() < targetSpeciesCount_ && compatThreshold_ > thresholdAdjustStep_)
            compatThreshold_ -= thresholdAdjustStep_;
    }

    // reset all species (but keep their historical best & reps)
    for (auto& s : species_) 
        s.resetForNextGen();

    // assign each genome to a species (or make a new one)
    for (auto* g : population_) {
        bool placed = false;
        for (auto& s : species_) {
            if (compatibilityDistance(*g, *s.representative) <= compatThreshold_) {
                s.members.push_back(g);
                placed = true;
                break;
            }
        }
        if (!placed) {
            Species newS;
            newS.representative = g;
            newS.members.push_back(g);
            species_.push_back(std::move(newS));
        }
    }

    // update each species: new representative, stagnation, possibly cull
    std::vector<Species> survivors;
    survivors.reserve(species_.size());
    for (auto& s : species_) {
        // 1) if it's empty, drop it immediately
        if (s.members.empty()) 
            continue;

        // 2) pick new rep
        s.representative = s.members.front();

        // 3) check stagnation
        float bestThisGen = 0;
        for (auto* g : s.members)
            bestThisGen = std::max(bestThisGen, g->fitness);

        if (bestThisGen > s.bestFitnessEver) {
            s.bestFitnessEver      = bestThisGen;
            s.gensSinceImprovement = 0;
        } else {
            s.gensSinceImprovement++;
        }

        // 4) only keep if still alive
        if (s.gensSinceImprovement <= stagnationLimit_)
            survivors.push_back(std::move(s));
    }
    species_.swap(survivors);
}

void NEAT::reproduce() {
    if (species_.empty()) {
    std::cerr << "No species to reproduce from! Skipping reproduce().\n";
    return;
}
    for (auto& s : species_) {
        s.adjustedFitnessSum = 0.0;
    }
    // 1) compute adjusted fitness & total
    double totalAdjusted = 0.0;
    for (auto& s : species_) {
        int m = s.members.size();
        for (auto* g : s.members) {
            double adj = g->fitness / double(m);
            s.adjustedFitnessSum += adj;
            totalAdjusted += adj;
        }
    }

    // 2) allocate offspring quotas
    int N = popSize_;
    std::vector<int> quotas(species_.size());
    int allocated = 0;
    for (size_t i = 0; i < species_.size(); ++i) {
        quotas[i] = int(std::round((species_[i].adjustedFitnessSum/totalAdjusted)*N));
        allocated += quotas[i];
    }
    // rebalance to exactly popSize_
    int diff = N - allocated;
    // sort species by adjusted sum desc
    std::vector<size_t> idx(species_.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
        [&](size_t a, size_t b){
            return species_[a].adjustedFitnessSum > species_[b].adjustedFitnessSum;
        });
    // // … after computing quotas and allocated …
    // std::cerr << "DEBUG: species_.size() = " << species_.size()
    //           << ", totalAdjusted = " << totalAdjusted << "\n";
    // int sumQ = 0;
    // for (size_t i = 0; i < quotas.size(); ++i) {
    //     std::cerr << "DEBUG: quotas[" << i << "] = " << quotas[i] << "\n";
    //     sumQ += quotas[i];
    // }
    // std::cerr << "DEBUG: sum of quotas = " << sumQ << " (should be " << N << ")\n";


    if (!idx.empty()) {
        for (int k = 0; diff != 0; ++k) {
            size_t i = idx[k % idx.size()];
            if (diff > 0) { quotas[i]++; diff--; }
            else if (quotas[i] > 1) { quotas[i]--; diff++; }
        }
    }

    // 3) build new population
    std::vector<Genome*> newPop;
    newPop.reserve(popSize_);
    std::uniform_real_distribution<float> uni(0,1);

    for (size_t i = 0; i < species_.size(); ++i) {
        auto& s = species_[i];
        int q = quotas[i];
        if (q <= 0) continue;

        // sort members by raw fitness descending
        std::sort(s.members.begin(), s.members.end(),
                  [](Genome* a, Genome* b){ return a->fitness > b->fitness; });


        // --- 3a) elitism: carry over the best ---
        // create a copy of the species’ best genome…
        Genome* repChild = new Genome(*s.members[0]);
        // …and immediately update the representative pointer so it never dangles:
        s.representative = repChild;
        newPop.push_back(repChild);
        q--;
        
        // --- 3b) fill the rest by intra‐species crossover+mutation ---
        std::vector<double> weights;
        weights.reserve(s.members.size());
        for (auto* g : s.members)
            weights.push_back(g->fitness / double(s.members.size()));
        std::discrete_distribution<size_t> pick(weights.begin(), weights.end());

        for (int j = 0; j < q; ++j) {
            Genome* p1 = s.members[pick(rng_)];
            Genome* p2 = s.members[pick(rng_)];
            if (p2->fitness > p1->fitness) std::swap(p1,p2);

            Genome* child = new Genome(Genome::crossover(*p1,*p2));
            child->mutateWeights();
            if (uni(rng_) < PROB_ADD_CONNECTION) child->mutateAddConnection();
            if (uni(rng_) < PROB_ADD_NODE)       child->mutateAddNode();
            newPop.push_back(child);
        }
    }

    // ── Drop any species that had quotas==0 ──
    {
      std::vector<Species> survivors;
      survivors.reserve(species_.size());
      for (size_t i = 0; i < species_.size(); ++i) {
        if (quotas[i] > 0) {
          survivors.push_back(std::move(species_[i]));
        }
      }
      species_.swap(survivors);
    }

    // --- 4) swap-and-delete old population in one clean move ---
    std::vector<Genome*> oldPop;
    oldPop.swap(population_);       // take ownership of the old pointers
    population_.swap(newPop);       // now population_ is the brand-new generation

    // now it's safe to delete the old generation, since nothing refers to them any more
    for (auto* g : oldPop) {
        delete g;
    }

    if (population_.size() != popSize_) {
        std::cerr << "ERROR: newPop.size() = " << population_.size() << " expected " << popSize_ << std::endl;
    }

}
