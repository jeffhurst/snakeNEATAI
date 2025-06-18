// InnovationTracker.h
#pragma once
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <string>

namespace neat {

// Forward declarations
using NodeId  = uint32_t;
using InnovId = uint64_t;

/**
 * @brief  Global, thread-safe innovation registry.
 * 
 * - Assigns unique innovation IDs to every new connection (from→to).
 * - Assigns unique node IDs when splitting existing connections, re-using
 *   the same node ID if the same connection is split again.
 * - Persists its state in a simple text file so IDs remain consistent
 *   across runs.
 */
class InnovationTracker {
public:
    /// Get the singleton instance
    static InnovationTracker& getInstance();

    /// Get (or create) the innovation number for a connection (from→to)
    InnovId getConnectionInnov(NodeId from, NodeId to);

    /// Get (or create) the new NodeId for splitting an existing connection
    /// whose innovation number is connInnov.
    NodeId getSplitNodeId(InnovId connInnov);

    /**
     * If your initial genomes define input IDs 0…inN-1 and output IDs
     * inN…inN+outN-1, call this *once* to ensure hidden-node IDs start
     * above that range.
     */
    void initializeNodeCounter(NodeId firstFreeId);

private:
    InnovationTracker();               // loads from disk
    ~InnovationTracker();              // saves to disk

    InnovationTracker(const InnovationTracker&)            = delete;
    InnovationTracker& operator=(const InnovationTracker&) = delete;

    mutable std::mutex mutex_;
    InnovId nextConnInnov_;
    NodeId  nextNodeId_;

    // key = (uint64_t(from)<<32)|uint32_t(to)
    std::unordered_map<uint64_t, InnovId> connInnovMap_;

    // key = original connection InnovId → NodeId of the split node
    std::unordered_map<InnovId, NodeId>   splitNodeMap_;

    const std::string dbFile_ = "innovation.db";
};

} // namespace neat
