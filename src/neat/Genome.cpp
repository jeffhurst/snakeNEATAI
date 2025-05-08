#include "Genome.h"
#include <random>
#include <algorithm>

namespace neat {

static std::mt19937 rng{std::random_device{}()};
static std::uniform_real_distribution<float> uni(-1.0f,1.0f);

void Genome::mutateWeights() {
    for (auto& kv : connections) {
        if (uni(rng) < 0.8f) // 80% perturb
            kv.second.weight += std::normal_distribution<float>(0,0.2f)(rng);
        else               // 20% assign new
            kv.second.weight = uni(rng);
    }
}

void Genome::mutateAddConnection() {
    // pick two nodes at random, ensure no existing connection, add new
    std::vector<NodeId> ids;
    for (auto& n : nodes) ids.push_back(n.first);
    std::uniform_int_distribution<size_t> di(0, ids.size()-1);
    for (int tries=0; tries<10; ++tries) {
        NodeId a = ids[di(rng)], b = ids[di(rng)];
        if (a==b) continue;
        // check existing
        bool exists = false;
        for (auto& kv: connections)
            if (kv.second.from==a && kv.second.to==b) { exists=true; break; }
        if (exists) continue;
        InnovId newInnov = /* global innovation counter stub */ a*100000 + b;
        connections[newInnov] = {newInnov,a,b,uni(rng),true};
        return;
    }
}

void Genome::mutateAddNode() {
    // pick a random connection to split
    if (connections.empty()) return;
    auto it = connections.begin();
    std::advance(it, std::uniform_int_distribution<size_t>(0,connections.size()-1)(rng));
    ConnectionGene cg = it->second;
    if (!cg.enabled) return;
    cg.enabled = false;
    InnovId in = it->first;
    // create new node
    NodeId newId = static_cast<NodeId>(nodes.size()+1);
    nodes[newId] = {newId, NodeGene::HIDDEN};
    // two new connections
    InnovId i1 = in*2+1, i2 = in*2+2;
    connections[in] = cg;
    connections[i1] = {i1, cg.from, newId, 1.0f, true};
    connections[i2] = {i2, newId, cg.to, cg.weight, true};
}

Genome Genome::crossover(const Genome& a, const Genome& b) {
    // assume a.fitness >= b.fitness
    Genome child;
    // copy all nodes
    child.nodes = a.nodes;
    // for each gene in a, pick from a or b if present
    for (auto& kv : a.connections) {
        auto itb = b.connections.find(kv.first);
        if (itb != b.connections.end() && uni(rng) < 0.5f)
            child.connections[kv.first] = itb->second;
        else
            child.connections[kv.first] = kv.second;
    }
    return child;
}

} // namespace neat
