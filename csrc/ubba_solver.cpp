#include "ubba_solver.hpp"
#include <chrono>
#include <numeric>
#include <iostream>

namespace kvmem {

namespace {

struct alignas(32) AdmissibleItem {
    float efficiency;     // weight / cost
    float weight;
    float rho;
    uint32_t cost;
    uint32_t block_index;
    uint16_t tier_index;
};

struct alignas(16) CompactLadderItem {
    float efficiency;
    float weight;
    uint32_t block_index;
    uint16_t cost;
    uint8_t tier_index;
    uint8_t dummy;
};

inline void radix_sort_items_descending(
    CompactLadderItem* in,
    CompactLadderItem* out,
    size_t n
) {
    if (n <= 1) return;

    uint32_t count0[256] = {0};
    uint32_t count1[256] = {0};
    uint32_t count2[256] = {0};
    uint32_t count3[256] = {0};

    for (size_t i = 0; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, &in[i].efficiency, sizeof(u));
        uint32_t key = ~u;
        count0[key & 0xFF]++;
        count1[(key >> 8) & 0xFF]++;
        count2[(key >> 16) & 0xFF]++;
        count3[(key >> 24) & 0xFF]++;
    }

    uint32_t pref0[256], pref1[256], pref2[256], pref3[256];
    pref0[0] = 0; pref1[0] = 0; pref2[0] = 0; pref3[0] = 0;
    for (int i = 1; i < 256; ++i) {
        pref0[i] = pref0[i - 1] + count0[i - 1];
        pref1[i] = pref1[i - 1] + count1[i - 1];
        pref2[i] = pref2[i - 1] + count2[i - 1];
        pref3[i] = pref3[i - 1] + count3[i - 1];
    }

    // Pass 0: in -> out
    for (size_t i = 0; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, &in[i].efficiency, sizeof(u));
        uint32_t b0 = (~u) & 0xFF;
        out[pref0[b0]++] = in[i];
    }

    // Pass 1: out -> in
    for (size_t i = 0; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, &out[i].efficiency, sizeof(u));
        uint32_t b1 = ((~u) >> 8) & 0xFF;
        in[pref1[b1]++] = out[i];
    }

    // Pass 2: in -> out
    for (size_t i = 0; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, &in[i].efficiency, sizeof(u));
        uint32_t b2 = ((~u) >> 16) & 0xFF;
        out[pref2[b2]++] = in[i];
    }

    // Pass 3: out -> in
    for (size_t i = 0; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, &out[i].efficiency, sizeof(u));
        uint32_t b3 = ((~u) >> 24) & 0xFF;
        in[pref3[b3]++] = out[i];
    }
}

} // anonymous namespace

UBBAResult solve_ubba_greedy(
    const std::vector<BlockCandidate>& blocks,
    double target_coverage,
    double rho_floor,
    bool enforce_fidelity_cliff,
    float s_scale
) {
    const auto t_start = std::chrono::high_resolution_clock::now();

    const double effective_rho_floor = enforce_fidelity_cliff
        ? std::min(rho_floor, LADDER_FIDELITY_CLIFF)
        : rho_floor;

    std::vector<AdmissibleItem> admissible_items;
    admissible_items.reserve(blocks.size());

    // Step 1: Fidelity Gating & Tier Selection per Block
    for (uint32_t b_idx = 0; b_idx < static_cast<uint32_t>(blocks.size()); ++b_idx) {
        const auto& blk = blocks[b_idx];
        
        int best_tier_idx = -1;
        uint32_t min_bytes = UINT32_MAX;
        float best_rho = 0.0f;

        for (uint16_t t_idx = 0; t_idx < static_cast<uint16_t>(blk.tier_profiles.size()); ++t_idx) {
            const auto& tp = blk.tier_profiles[t_idx];
            if (tp.distortion_rho <= effective_rho_floor) {
                if (tp.bytes_per_block < min_bytes) {
                    min_bytes = tp.bytes_per_block;
                    best_tier_idx = t_idx;
                    best_rho = tp.distortion_rho;
                }
            }
        }

        if (best_tier_idx >= 0) {
            const float weight = std::max(blk.weight, 1e-12f);
            const float eff = weight / static_cast<float>(min_bytes);
            admissible_items.push_back({
                eff,
                weight,
                best_rho,
                min_bytes,
                b_idx,
                static_cast<uint16_t>(best_tier_idx)
            });
        }
    }

    // Step 2: Sort by efficiency descending (highest coverage per byte first)
    std::sort(admissible_items.begin(), admissible_items.end(),
              [](const AdmissibleItem& a, const AdmissibleItem& b) noexcept {
                  return a.efficiency > b.efficiency;
              });

    // Step 3: Greedy Accumulation
    UBBAResult result;
    result.rho_floor = effective_rho_floor;
    result.target_coverage = target_coverage;
    result.total_blocks_considered = blocks.size();

    uint64_t total_bytes = 0;
    double cum_weight = 0.0;
    double sum_rho = 0.0;
    double max_rho = 0.0;
    double shadow_price = 0.0;
    uint32_t marginal_cost = 0;

    result.allocations.reserve(admissible_items.size());

    for (const auto& item : admissible_items) {
        const auto& blk = blocks[item.block_index];
        const auto& tp = blk.tier_profiles[item.tier_index];

        AllocationDecision decision;
        decision.block_id = blk.block_id;
        decision.block_name = blk.block_name;
        decision.tier_name = tp.tier_name;
        decision.tier_type = tp.type;
        decision.bytes_per_block = tp.bytes_per_block;
        decision.distortion_rho = tp.distortion_rho;
        decision.tier_bias = tp.get_tier_bias(s_scale);

        result.allocations.push_back(std::move(decision));

        total_bytes += item.cost;
        cum_weight += item.weight;
        sum_rho += item.rho;
        if (item.rho > max_rho) {
            max_rho = item.rho;
        }

        shadow_price = (item.efficiency > 0.0f) ? (1.0 / item.efficiency) : 0.0;
        marginal_cost = item.cost;

        if (cum_weight >= target_coverage) {
            break;
        }
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    const double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

    result.total_bytes = total_bytes;
    result.achieved_coverage = cum_weight;
    result.max_rho = max_rho;
    result.num_blocks_selected = result.allocations.size();
    result.mean_rho = result.num_blocks_selected > 0 ? (sum_rho / result.num_blocks_selected) : 0.0;
    result.shadow_price_lambda = shadow_price;
    result.duality_gap = static_cast<double>(marginal_cost);
    result.solver_latency_us = elapsed_us;
    result.solver_latency_ms = elapsed_us / 1000.0;
    result.is_feasible = (cum_weight >= target_coverage);

    return result;
}

UBBAResult solve_ubba_ladder_fast(
    const std::vector<FastBlockCandidate>& blocks,
    double target_coverage,
    double rho_floor,
    bool enforce_fidelity_cliff,
    float s_scale
) {
    const auto t_start = std::chrono::high_resolution_clock::now();

    const double effective_rho_floor = enforce_fidelity_cliff
        ? std::min(rho_floor, LADDER_FIDELITY_CLIFF)
        : rho_floor;
    const float eff_rho_f = static_cast<float>(effective_rho_floor);

    thread_local std::vector<CompactLadderItem> tls_items_in;
    thread_local std::vector<CompactLadderItem> tls_items_out;
    tls_items_in.clear();
    if (tls_items_in.capacity() < blocks.size()) {
        tls_items_in.reserve(blocks.size());
    }
    if (tls_items_out.size() < blocks.size()) {
        tls_items_out.resize(blocks.size());
    }

    // Step 1: Branchless / Fast ordered tier check from MERGED (idx 3) to FP8 (idx 0)
    // Tiers are strictly ordered by bytes: MERGED(128) < INT2(256) < INT4(512) < FP8(1024)
    for (uint32_t b_idx = 0; b_idx < static_cast<uint32_t>(blocks.size()); ++b_idx) {
        const auto& blk = blocks[b_idx];

        int best_tier = -1;
        // Priority order: 3 (MERGED), 2 (INT2), 1 (INT4), 0 (FP8)
        if (blk.rhos[3] <= eff_rho_f) {
            best_tier = 3;
        } else if (blk.rhos[2] <= eff_rho_f) {
            best_tier = 2;
        } else if (blk.rhos[1] <= eff_rho_f) {
            best_tier = 1;
        } else if (blk.rhos[0] <= eff_rho_f) {
            best_tier = 0;
        }

        if (best_tier >= 0) {
            const uint16_t cost = static_cast<uint16_t>(blk.bytes[best_tier]);
            const float weight = std::max(blk.weight, 1e-12f);
            const float eff = weight / static_cast<float>(cost);

            tls_items_in.push_back({
                eff,
                weight,
                b_idx,
                cost,
                static_cast<uint8_t>(best_tier),
                0
            });
        }
    }

    // Step 2: Ultra-fast 4-pass radix sort (LSD descending) - ~15 microseconds
    radix_sort_items_descending(tls_items_in.data(), tls_items_out.data(), tls_items_in.size());
    const auto& admissible_items = tls_items_in;

    // Step 3: Greedy demand accumulation
    UBBAResult result;
    result.rho_floor = effective_rho_floor;
    result.target_coverage = target_coverage;
    result.total_blocks_considered = blocks.size();

    uint64_t total_bytes = 0;
    double cum_weight = 0.0;
    double sum_rho = 0.0;
    double max_rho = 0.0;
    double shadow_price = 0.0;
    uint32_t marginal_cost = 0;

    static const char* const kTierNames[4] = {"FP8", "INT4", "INT2", "MERGED"};
    static const TierType kTierTypes[4] = {TierType::FP8, TierType::INT4, TierType::INT2, TierType::MERGED};

    result.allocations.resize(admissible_items.size());
    AllocationDecision* alloc_ptr = result.allocations.data();
    size_t count = 0;

    for (const auto& item : admissible_items) {
        const float rho = blocks[item.block_index].rhos[item.tier_index];
        const float sigma = rho * s_scale;

        AllocationDecision& dec = alloc_ptr[count++];
        dec.block_id = blocks[item.block_index].block_id;
        dec.block_name = std::string_view();
        dec.tier_name = kTierNames[item.tier_index];
        dec.tier_type = kTierTypes[item.tier_index];
        dec.bytes_per_block = item.cost;
        dec.distortion_rho = rho;
        dec.tier_bias = -(sigma * sigma) * 0.5f;

        total_bytes += item.cost;
        cum_weight += item.weight;
        sum_rho += rho;
        if (rho > max_rho) {
            max_rho = rho;
        }

        if (cum_weight >= target_coverage) {
            shadow_price = (item.efficiency > 0.0f) ? (1.0 / static_cast<double>(item.efficiency)) : 0.0;
            marginal_cost = item.cost;
            break;
        }
    }
    result.allocations.resize(count);

    const auto t_end = std::chrono::high_resolution_clock::now();
    const double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

    result.total_bytes = total_bytes;
    result.achieved_coverage = cum_weight;
    result.max_rho = max_rho;
    result.num_blocks_selected = result.allocations.size();
    result.mean_rho = result.num_blocks_selected > 0 ? (sum_rho / result.num_blocks_selected) : 0.0;
    result.shadow_price_lambda = shadow_price;
    result.duality_gap = static_cast<double>(marginal_cost);
    result.solver_latency_us = elapsed_us;
    result.solver_latency_ms = elapsed_us / 1000.0;
    result.is_feasible = (cum_weight >= target_coverage);

    return result;
}

BlockCandidate create_ladder_block_candidate(
    uint32_t id,
    const std::string& name,
    float weight,
    float dispersity,
    uint32_t base_bytes
) {
    BlockCandidate blk(id, name, weight, dispersity);
    blk.add_tier("FP8", base_bytes, 0.010f, TierType::FP8);
    blk.add_tier("INT4", base_bytes / 2, 0.120f * (1.0f + 0.5f * dispersity), TierType::INT4);
    blk.add_tier("INT2", base_bytes / 4, 0.343f * (1.0f + 0.3f * dispersity), TierType::INT2);
    blk.add_tier("MERGED", base_bytes / 8, 0.420f * (1.0f + 0.2f * dispersity), TierType::MERGED);
    return blk;
}

FastBlockCandidate create_fast_ladder_candidate(
    uint32_t id,
    float weight,
    float dispersity,
    uint32_t base_bytes
) {
    FastBlockCandidate blk;
    blk.block_id = id;
    blk.weight = weight;
    blk.dispersity = dispersity;

    blk.bytes[0] = base_bytes;
    blk.bytes[1] = base_bytes / 2;
    blk.bytes[2] = base_bytes / 4;
    blk.bytes[3] = base_bytes / 8;

    blk.rhos[0] = 0.010f;
    blk.rhos[1] = 0.120f * (1.0f + 0.5f * dispersity);
    blk.rhos[2] = 0.343f * (1.0f + 0.3f * dispersity);
    blk.rhos[3] = 0.420f * (1.0f + 0.2f * dispersity);

    return blk;
}

} // namespace kvmem
