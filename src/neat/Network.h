// Network.h
#pragma once
#include "Gene.h"
#include "Genome.h"
#include <vector>
#include <unordered_map> 

namespace neat {

class Network {
public:
    explicit Network(const Genome& g);
    // Feedforward: inputs → outputs; records per-node activations
    std::vector<float> feed(const std::vector<float>& in);

    // Access the last activation values by node ID
    const std::unordered_map<NodeId, float>& getActivations() const { return activations_; }

    // Access genome for structure
    const Genome& getGenome() const { return genome_; }

private:
    const Genome& genome_;
    std::vector<NodeId> topoOrder_;
    std::unordered_map<NodeId, float> activations_;  // recorded after feed
    void buildTopology();
};

} // namespace neat
