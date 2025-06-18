// Genome.cpp
#include "Genome.h"
#include "InnovationTracker.h"
#include "NeatConfig.h"
#include <random>
#include <algorithm>
#include <set>
#include <iostream>

namespace neat {

static std::mt19937 rng{std::random_device{}()};
static std::uniform_real_distribution<float> uni(-1.0f,1.0f);

void Genome::mutateWeights() {
    static std::normal_distribution<float> perturbDist(0.0f, PERTURB_STRENGTH);
    for (auto& kv : connections) {
        float r = uni(rng);
        if (r < WEIGHT_PERTURB_PROB) {
            // tweak existing weight
            kv.second.weight += perturbDist(rng);
        } else {
            // assign new weight
            kv.second.weight  = uni(rng);
        }
    }
}

void Genome::mutateAddConnection() {
    // gather all node IDs
    std::vector<NodeId> ids;
    ids.reserve(nodes.size());
    for (auto& kv : nodes) ids.push_back(kv.first);

    std::uniform_int_distribution<size_t> di(0, ids.size()-1);
    for (int tries = 0; tries < 10; ++tries) {
        NodeId a = ids[di(rng)], b = ids[di(rng)];
        if (a == b) continue;
        // never connect *into* an input
        if (nodes[b].type == NodeGene::INPUT || nodes[b].type == NodeGene::BIAS) continue;
        // skip existing
        bool exists = false;
        for (auto& ck : connections)
            if (ck.second.from == a && ck.second.to == b) { exists = true; break; }
        if (exists) continue;

        InnovId innov = InnovationTracker::getInstance().getConnectionInnov(a, b);
        connections[innov] = { innov, a, b, uni(rng), true };
        return;
    }
}


void Genome::mutateAddNode() {
    if (connections.empty()) return;

    // pick a random enabled connection
    auto it = connections.begin();
    std::advance(it, std::uniform_int_distribution<size_t>(0, connections.size()-1)(rng));
    ConnectionGene cg = it->second;
    if (!cg.enabled) return;

    // disable the old link
    connections[cg.innov].enabled = false;

    // fetch or create the new hidden node ID
    NodeId newId = InnovationTracker::getInstance().getSplitNodeId(cg.innov);
    nodes[newId] = { newId, NodeGene::HIDDEN };

    // create two new connections: from→new, new→to
    InnovId in1 = InnovationTracker::getInstance().getConnectionInnov(cg.from, newId);
    InnovId in2 = InnovationTracker::getInstance().getConnectionInnov(newId,   cg.to);
    connections[in1] = { in1, cg.from, newId,   1.0f,      true };
    connections[in2] = { in2,   newId,   cg.to,   cg.weight, true };
}


Genome Genome::crossover(const Genome& g1, const Genome& g2) {
    // Determine fitter parent (or random if tie)
    const Genome *fit, *oth;
    if      (g1.fitness > g2.fitness) { fit = &g1; oth = &g2; }
    else if (g2.fitness > g1.fitness) { fit = &g2; oth = &g1; }
    else {  // tie: pick randomly
        if (uni(rng) < 0.5f) { fit = &g1; oth = &g2; }
        else                 { fit = &g2; oth = &g1; }
    }

    Genome child;
    // 1) copy all node genes from fitter parent
    child.nodes = fit->nodes;

    // 2) gather all innovation IDs
    std::set<InnovId> allInnov;
    for (auto& kv : fit->connections) allInnov.insert(kv.first);
    for (auto& kv : oth->connections) allInnov.insert(kv.first);

    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    // 3) for each gene ID, decide inheritance
    for (InnovId innov : allInnov) {
        auto itF = fit->connections.find(innov);
        auto itO = oth->connections.find(innov);

        if (itF != fit->connections.end() && itO != oth->connections.end()) {
            // matching gene: pick randomly
            const ConnectionGene *src =
                (coin(rng) < 0.5f ? &itF->second : &itO->second);
            child.connections[innov] = *src;

            // handle disabled → re-enable chance
            if (!itF->second.enabled || !itO->second.enabled) {
                bool enable = (coin(rng) < PROB_REENABLE_GENE);
                child.connections[innov].enabled = true;
            }

        } else if (itF != fit->connections.end()) {
            // disjoint or excess from fitter parent
            child.connections[innov] = itF->second;
        }
        // else: gene only in less-fit parent → skip
    }
    return child;
}

} // namespace neat
