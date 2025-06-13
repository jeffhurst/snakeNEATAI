// Genome.h
#pragma once
#include "Gene.h"
#include <vector>
#include <map>

namespace neat {

struct Genome {
    std::map<InnovId, ConnectionGene> connections;
    std::map<NodeId, NodeGene> nodes;
    float fitness = 0.0f;
    // mutation/crossover APIs
    void mutateAddConnection();
    void mutateAddNode();
    void mutateWeights();
    static Genome crossover(const Genome& a, const Genome& b);
};

} // namespace neat
