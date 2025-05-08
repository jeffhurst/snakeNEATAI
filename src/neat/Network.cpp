#include "Network.h"
#include <unordered_map>
#include <queue>
#include <cmath>
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
    // assign inputs
    for (auto& kv : genome_.nodes) {
        if (kv.second.type == NodeGene::INPUT)
            values[kv.first] = in.at(i++);
        else
            values[kv.first] = 0.0f;
    }
    // propagate
    for (NodeId nid : topoOrder_) {
        float v = values[nid];
        for (auto& kv : genome_.connections) if (kv.second.enabled && kv.second.from==nid) {
            values[kv.second.to] += v * kv.second.weight;
        }
        // activation
        if (genome_.nodes.at(nid).type != NodeGene::INPUT)
            values[nid] = std::tanh(values[nid]);
    }
    // collect outputs
    std::vector<float> out;
    for (auto& kv : genome_.nodes)
        if (kv.second.type == NodeGene::OUTPUT)
            out.push_back(values[kv.first]);
    return out;
}
