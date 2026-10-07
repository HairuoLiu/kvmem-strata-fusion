#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include <cmath>

namespace kvmem {

struct LSEResult {
    float lse;
    float z_normalizer;
};

struct DoubletSplitResult {
    bool is_split;
    size_t needle_token_idx;
    float max_dispersion;
    float avg_dispersion;
    std::vector<float> mean_centroid;      // [dim]
    std::vector<float> needle_centroid;    // [dim] (if split)
    std::vector<float> residual_centroid;  // [dim] (if split)
};

class LSEEngine {
public:
    explicit LSEEngine(size_t head_dim = 64);

    // Compute dot-product logits: logits[j] = (query . candidates[j]) / sqrt(dim)
    // Accelerated with ARM Neon SIMD when available
    void compute_logits(
        const float* query,
        const float* candidates,
        size_t num_candidates,
        float* out_logits
    ) const;

    // Numerically stable Log-Sum-Exp computation
    LSEResult compute_lse(const float* logits, size_t n) const;

    // Compute discarded attention mass: rho = 1 - sum_{j in selected} exp(s_j) / Z
    float compute_discarded_mass(
        const float* full_logits,
        size_t n_full,
        const size_t* selected_indices,
        size_t n_selected,
        float full_lse
    ) const;

    // Rank top-k candidates using unnormalized logits
    void rank_unnormalized(
        const float* logits,
        size_t n,
        size_t top_k,
        std::vector<size_t>& out_indices
    ) const;

    size_t head_dim() const { return head_dim_; }
    float scale() const { return scale_; }

private:
    size_t head_dim_;
    float scale_;
};

class AntiCollapseDispersionTracker {
public:
    explicit AntiCollapseDispersionTracker(
        float dispersion_threshold = 0.85f,
        size_t head_dim = 64
    );

    // Deroped mean and dispersion computation for a block of tokens
    // keys: [num_tokens * head_dim] row-major
    void compute_deroped_mean(
        const float* keys,
        size_t num_tokens,
        float* out_mean,
        float& out_max_dispersion,
        float& out_avg_dispersion
    ) const;

    // Inspect block and perform doublet splitting if max_dispersion > threshold
    DoubletSplitResult inspect_and_split(
        const float* keys,
        size_t num_tokens
    ) const;

    float threshold() const { return dispersion_threshold_; }
    size_t head_dim() const { return head_dim_; }

private:
    float dispersion_threshold_;
    size_t head_dim_;
};

} // namespace kvmem
