#pragma once
#include <cstdint>

namespace neat {

using NodeId = uint32_t;
using InnovId = uint64_t;

// NodeGene: id and type (input, hidden, output)
struct NodeGene {
    NodeId id;
    enum Type { INPUT, HIDDEN, OUTPUT } type;
};

// ConnectionGene: from, to, weight, enabled, innovation number
struct ConnectionGene {
    InnovId innov;
    NodeId from, to;
    float weight;
    bool enabled;
};

} // namespace neat
