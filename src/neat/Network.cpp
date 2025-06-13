// Network.cpp
#include "Network.h"
#include <unordered_map>
#include <queue>
#include <cmath>
#include <iostream>
using namespace neat;

Network::Network(const Genome& g)
 : genome_(g)
{
    buildTopology();
}

void Network::buildTopology() {
    // simple Kahn’s algorithm: nodes with no incoming edges first
    std::unordered_map<NodeId,int> indeg;
    for (auto& kv : genome_.nodes) indeg[kv.first] = 0;
    for (auto& kv : genome_.connections) if (kv.second.enabled)
        indeg[kv.second.to]++;
    std::queue<NodeId> q;
    for (auto& kv : indeg) if (kv.second==0) q.push(kv.first);
    while (!q.empty()) {
        NodeId n = q.front(); q.pop();
        topoOrder_.push_back(n);
        for (auto& kv : genome_.connections) {
            auto& cg = kv.second;
            if (cg.enabled && cg.from==n) {
                if (--indeg[cg.to]==0) q.push(cg.to);
            }
        }
    }
}

std::vector<float> Network::feed(const std::vector<float>& in) {
    std::unordered_map<NodeId,float> values;
    size_t i = 0;
    // 1) initialize all node values
    for (auto& kv : genome_.nodes) {
        if (kv.second.type == NodeGene::INPUT)
            values[kv.first] = in.at(i++);
        else
            values[kv.first] = 0.0f;
    }

    // 2) propagate in topological order
    for (NodeId nid : topoOrder_) {
        float v = values[nid];
        for (auto& ck : genome_.connections) {
            auto& cg = ck.second;
            if (cg.enabled && cg.from == nid) {
                values[cg.to] += v * cg.weight;
            }
        }

        // 3) activation — guard the at()
        auto it = genome_.nodes.find(nid);
        if (it != genome_.nodes.end()) {
            if (it->second.type != NodeGene::INPUT) {
                values[nid] = std::tanh(values[nid]);
            }
        } else {
            // pass
        }
    }
    activations_ = values;

    // 4) collect outputs
    std::vector<float> out;
    for (auto& kv : genome_.nodes) {
        if (kv.second.type == NodeGene::OUTPUT) {
            out.push_back(values[kv.first]);
        }
    }
    return out;
}
