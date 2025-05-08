#pragma once
#include "Gene.h"
#include "Genome.h"
#include <vector>

namespace neat {

class Network {
public:
    explicit Network(const Genome& g);
    // feedforward: inputs → outputs
    std::vector<float> feed(const std::vector<float>& in);
private:
    const Genome& genome_;
    // cached topological order
    std::vector<NodeId> topoOrder_;
    void buildTopology();
};

} // namespace neat
