// InnovationTracker.cpp
#include "InnovationTracker.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace neat {

InnovationTracker& InnovationTracker::getInstance() {
    static InnovationTracker inst;
    return inst;
}

InnovationTracker::InnovationTracker()
  : nextConnInnov_(1),   // start at 1
    nextNodeId_(1)       // will be bumped by initializeNodeCounter()
{
    std::lock_guard<std::mutex> lk(mutex_);
    std::ifstream in(dbFile_);
    if (!in.is_open()) return;  // first run: no file yet

    size_t connCount, splitCount;
    in >> nextConnInnov_ >> nextNodeId_;
    in >> connCount;
    for (size_t i = 0; i < connCount; ++i) {
        uint64_t key; InnovId innov;
        in >> key >> innov;
        connInnovMap_[key] = innov;
    }
    in >> splitCount;
    for (size_t i = 0; i < splitCount; ++i) {
        InnovId connInnov; NodeId nid;
        in >> connInnov >> nid;
        splitNodeMap_[connInnov] = nid;
    }
}

InnovationTracker::~InnovationTracker() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ofstream out(dbFile_, std::ofstream::trunc);
    if (!out.is_open()) {
        std::cerr << "Failed to save innovation DB to “" << dbFile_ << "”\n";
        return;
    }
    // save counters
    out << nextConnInnov_ << " " << nextNodeId_ << "\n";
    // save connection map
    out << connInnovMap_.size() << "\n";
    for (auto& kv : connInnovMap_) {
        out << kv.first << " " << kv.second << "\n";
    }
    // save split-node map
    out << splitNodeMap_.size() << "\n";
    for (auto& kv : splitNodeMap_) {
        out << kv.first << " " << kv.second << "\n";
    }
}

InnovId InnovationTracker::getConnectionInnov(NodeId from, NodeId to) {
    std::lock_guard<std::mutex> lk(mutex_);
    uint64_t key = (uint64_t(from) << 32) | uint64_t(to);
    auto it = connInnovMap_.find(key);
    if (it != connInnovMap_.end()) return it->second;
    InnovId innov = nextConnInnov_++;
    connInnovMap_[key] = innov;
    return innov;
}

NodeId InnovationTracker::getSplitNodeId(InnovId connInnov) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = splitNodeMap_.find(connInnov);
    if (it != splitNodeMap_.end()) return it->second;
    NodeId nid = nextNodeId_++;
    splitNodeMap_[connInnov] = nid;
    return nid;
}

void InnovationTracker::initializeNodeCounter(NodeId firstFreeId) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (nextNodeId_ < firstFreeId) {
        nextNodeId_ = firstFreeId;
    }
}

} // namespace neat
