// NeatConfig.h
#pragma once

namespace neat {

// Weight mutation:
//  - PROB_PERTURB:    % of weights to perturb vs. replace entirely
//  - PERTURB_STRENGTH: standard deviation of Gaussian for perturbation
constexpr float WEIGHT_PERTURB_PROB   = 0.8f;
constexpr float PERTURB_STRENGTH      = 0.2f;

// Structural mutation probabilities (used in NEAT::reproduce())
constexpr float PROB_ADD_CONNECTION   = 0.03f;
constexpr float PROB_ADD_NODE         = 0.01f;

// When a matching gene is disabled in either parent, re-enable it with this chance
constexpr float PROB_REENABLE_GENE    = 0.05f;

} // namespace neat
