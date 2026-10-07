#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

namespace kvmem {

// LADDER hard fidelity cliff boundary (rho <= 0.365)
constexpr double LADDER_FIDELITY_CLIFF = 0.365;

// Standard compression tier definitions
enum class TierType : uint8_t {
    FP8 = 0,
    INT4 = 1,
    INT2 = 2,
    MERGED = 3,
    CUSTOM = 4
};

inline const char* tier_type_to_string(TierType t) {
    switch (t) {
        case TierType::FP8: return "FP8";
        case TierType::INT4: return "INT4";
        case TierType::INT2: return "INT2";
        case TierType::MERGED: return "MERGED";
        default: return "CUSTOM";
    }
}

struct TierProfile {
    std::string tier_name;
    uint32_t bytes_per_block;
    float distortion_rho;
    TierType type;

    TierProfile() 
        : tier_name(""), bytes_per_block(0), distortion_rho(0.0f), type(TierType::CUSTOM) {}

    TierProfile(std::string name, uint32_t bytes, float rho, TierType t = TierType::CUSTOM)
        : tier_name(std::move(name)), bytes_per_block(bytes), distortion_rho(rho), type(t) {}

    inline float get_tier_bias(float s = 1.85f) const {
        float sigma = distortion_rho * s;
        return -(sigma * sigma) * 0.5f;
    }
};

struct BlockCandidate {
    uint32_t block_id;
    std::string block_name;
    float weight;
    float dispersity;
    std::vector<TierProfile> tier_profiles;

    BlockCandidate() : block_id(0), block_name(""), weight(0.0f), dispersity(0.0f) {}

    BlockCandidate(uint32_t id, std::string name, float w, float disp = 0.0f)
        : block_id(id), block_name(std::move(name)), weight(w), dispersity(disp) {}

    void add_tier(std::string name, uint32_t bytes, float rho, TierType t = TierType::CUSTOM) {
        tier_profiles.emplace_back(std::move(name), bytes, rho, t);
    }
};

// Compact POD candidate structure for maximum cache locality during 10K+ batch solving
struct alignas(32) FastBlockCandidate {
    uint32_t block_id;
    float weight;
    float dispersity;
    // Pre-computed tier parameters or generated on-the-fly
    uint32_t bytes[4];  // FP8, INT4, INT2, MERGED
    float rhos[4];
};

struct AllocationDecision {
    uint32_t block_id;
    std::string_view block_name;
    std::string_view tier_name;
    TierType tier_type;
    uint32_t bytes_per_block;
    float distortion_rho;
    float tier_bias;
};

struct UBBAResult {
    std::vector<AllocationDecision> allocations;
    uint64_t total_bytes;
    double achieved_coverage;
    double target_coverage;
    double rho_floor;
    double max_rho;
    double mean_rho;
    size_t num_blocks_selected;
    size_t total_blocks_considered;
    double shadow_price_lambda;
    double duality_gap;
    double solver_latency_us;  // Execution time in microseconds
    double solver_latency_ms;  // Execution time in milliseconds
    bool is_feasible;

    UBBAResult()
        : total_bytes(0),
          achieved_coverage(0.0),
          target_coverage(0.0),
          rho_floor(0.0),
          max_rho(0.0),
          mean_rho(0.0),
          num_blocks_selected(0),
          total_blocks_considered(0),
          shadow_price_lambda(0.0),
          duality_gap(0.0),
          solver_latency_us(0.0),
          solver_latency_ms(0.0),
          is_feasible(false) {}
};

// High-level API for arbitrary candidate vectors
UBBAResult solve_ubba_greedy(
    const std::vector<BlockCandidate>& blocks,
    double target_coverage,
    double rho_floor = 0.10,
    bool enforce_fidelity_cliff = true,
    float s_scale = 1.85f
);

// Ultra-fast zero-allocation path for LADDER blocks (target < 0.1ms for 10K blocks)
UBBAResult solve_ubba_ladder_fast(
    const std::vector<FastBlockCandidate>& blocks,
    double target_coverage,
    double rho_floor = 0.10,
    bool enforce_fidelity_cliff = true,
    float s_scale = 1.85f
);

// Helper function to create synthetic LADDER blocks with exact cost model
BlockCandidate create_ladder_block_candidate(
    uint32_t id,
    const std::string& name,
    float weight,
    float dispersity = 0.0f,
    uint32_t base_bytes = 1024
);

FastBlockCandidate create_fast_ladder_candidate(
    uint32_t id,
    float weight,
    float dispersity = 0.0f,
    uint32_t base_bytes = 1024
);

} // namespace kvmem
