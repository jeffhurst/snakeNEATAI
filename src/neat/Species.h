// Species.h
#pragma once
#include "Genome.h"
#include <vector>

namespace neat {

struct Species {
    std::vector<Genome*> members;
    Genome* best = nullptr;
    void reset() { members.clear(); best = nullptr; }
};

} // namespace neat