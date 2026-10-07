#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <numeric>
#include <cmath>
#include <cassert>

#include "ubba_solver.hpp"
#include "ifr_lse.hpp"
#include "casa_paged_table.hpp"

using namespace kvmem;

struct BenchStats {
    double min_us;
    double max_us;
    double mean_us;
    double median_us;
    double p95_us;
};

BenchStats compute_stats(std::vector<double>& latencies) {
    std::sort(latencies.begin(), latencies.end());
    const size_t n = latencies.size();
    const double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    const double mean = sum / n;
    const double min_val = latencies.front();
    const double max_val = latencies.back();
    const double median = (n % 2 == 0) ? (latencies[n / 2 - 1] + latencies[n / 2]) * 0.5 : latencies[n / 2];
    const size_t p95_idx = static_cast<size_t>(0.95 * n);
    const double p95 = latencies[std::min(p95_idx, n - 1)];
    return {min_val, max_val, mean, median, p95};
}

void print_banner(const std::string& title) {
    std::cout << "\n" << std::string(75, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(75, '=') << "\n";
}

// -----------------------------------------------------------------------------
// Benchmark 1: UBBA Knapsack Solver Latency (< 0.1ms target)
// -----------------------------------------------------------------------------
void bench_ubba_solver(size_t num_blocks = 10000, size_t iterations = 100) {
    print_banner("1. UBBA Solver Benchmark (Target < 0.1ms for 10K blocks)");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> w_dist(0.1f, 2.0f);
    std::uniform_real_distribution<float> disp_dist(0.05f, 0.40f);
    std::uniform_real_distribution<float> needle_dist(0.0f, 1.0f);

    std::vector<BlockCandidate> general_blocks;
    std::vector<FastBlockCandidate> fast_blocks;
    general_blocks.reserve(num_blocks);
    fast_blocks.reserve(num_blocks);

    double total_weight = 0.0;
    for (size_t i = 0; i < num_blocks; ++i) {
        const bool is_needle = (needle_dist(rng) < 0.10f);
        const float w = is_needle ? (w_dist(rng) * 10.0f) : w_dist(rng);
        const float disp = is_needle ? 1.2f : disp_dist(rng);
        total_weight += w;

        general_blocks.push_back(create_ladder_block_candidate(
            static_cast<uint32_t>(i), "blk_" + std::to_string(i), w, disp, 1024
        ));
        fast_blocks.push_back(create_fast_ladder_candidate(
            static_cast<uint32_t>(i), w, disp, 1024
        ));
    }

    const double target_coverage = total_weight * 0.70;
    const double rho_floor = 0.10;

    std::cout << "Configuration: " << num_blocks << " blocks, target coverage = " 
              << std::fixed << std::setprecision(2) << target_coverage << " (70%), rho_floor = " << rho_floor << "\n";
    std::cout << "Running " << iterations << " benchmark iterations...\n\n";

    // Bench A: General Object API
    std::vector<double> general_latencies;
    general_latencies.reserve(iterations);
    UBBAResult last_gen_res;

    for (size_t it = 0; it < iterations; ++it) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        last_gen_res = solve_ubba_greedy(general_blocks, target_coverage, rho_floor, true, 1.85f);
        const auto t1 = std::chrono::high_resolution_clock::now();
        general_latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    const auto gen_stats = compute_stats(general_latencies);
    std::cout << ">>> UBBA General API (BlockCandidate vector):\n";
    std::cout << "    Mean:   " << std::setw(7) << gen_stats.mean_us << " us (" << (gen_stats.mean_us / 1000.0) << " ms)\n";
    std::cout << "    Median: " << std::setw(7) << gen_stats.median_us << " us\n";
    std::cout << "    Min:    " << std::setw(7) << gen_stats.min_us << " us\n";
    std::cout << "    Max:    " << std::setw(7) << gen_stats.max_us << " us\n";
    std::cout << "    P95:    " << std::setw(7) << gen_stats.p95_us << " us\n";
    std::cout << "    Feasible: " << (last_gen_res.is_feasible ? "YES" : "NO")
              << " | Blocks selected: " << last_gen_res.num_blocks_selected << "/" << num_blocks
              << " | Max rho: " << last_gen_res.max_rho << " <= " << last_gen_res.rho_floor << "\n\n";

    // Bench B: Ultra-Fast Path (FastBlockCandidate POD)
    std::vector<double> fast_latencies;
    fast_latencies.reserve(iterations);
    UBBAResult last_fast_res;

    for (size_t it = 0; it < iterations; ++it) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        last_fast_res = solve_ubba_ladder_fast(fast_blocks, target_coverage, rho_floor, true, 1.85f);
        const auto t1 = std::chrono::high_resolution_clock::now();
        fast_latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    const auto fast_stats = compute_stats(fast_latencies);
    std::cout << ">>> UBBA Fast Path (FastBlockCandidate POD):\n";
    std::cout << "    Mean:   " << std::setw(7) << fast_stats.mean_us << " us (" << (fast_stats.mean_us / 1000.0) << " ms)\n";
    std::cout << "    Median: " << std::setw(7) << fast_stats.median_us << " us\n";
    std::cout << "    Min:    " << std::setw(7) << fast_stats.min_us << " us\n";
    std::cout << "    Max:    " << std::setw(7) << fast_stats.max_us << " us\n";
    std::cout << "    P95:    " << std::setw(7) << fast_stats.p95_us << " us\n";
    std::cout << "    Feasible: " << (last_fast_res.is_feasible ? "YES" : "NO")
              << " | Bytes allocated: " << last_fast_res.total_bytes / 1024 << " KB"
              << " | Shadow Price: " << last_fast_res.shadow_price_lambda << "\n";

    const bool target_met = (fast_stats.mean_us < 100.0);
    std::cout << "\n>>> LATENCY SLA STATUS (< 0.1ms / 100us for 10K blocks): "
              << (target_met ? "PASSED [OK]" : "FAILED")
              << " (" << fast_stats.mean_us << " us = " << (fast_stats.mean_us / 1000.0) << " ms)\n";
}

// -----------------------------------------------------------------------------
// Benchmark 2: Scaling Sweep across Block Counts
// -----------------------------------------------------------------------------
void bench_ubba_sweep() {
    print_banner("2. UBBA Scaling Sweep (1K -> 50K Blocks)");
    std::cout << std::left << std::setw(12) << "Blocks"
              << std::setw(16) << "Latency (us)"
              << std::setw(16) << "Latency (ms)"
              << std::setw(16) << "Selected"
              << std::setw(16) << "Throughput (M/s)" << "\n";
    std::cout << std::string(75, '-') << "\n";

    const std::vector<size_t> test_sizes = {1000, 2000, 5000, 10000, 20000, 50000};
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> w_dist(0.1f, 2.0f);

    for (size_t n : test_sizes) {
        std::vector<FastBlockCandidate> blocks;
        blocks.reserve(n);
        double total_w = 0.0;
        for (size_t i = 0; i < n; ++i) {
            float w = w_dist(rng);
            total_w += w;
            blocks.push_back(create_fast_ladder_candidate(static_cast<uint32_t>(i), w, 0.1f, 1024));
        }

        const size_t iters = (n <= 10000) ? 50 : 20;
        std::vector<double> lats;
        lats.reserve(iters);
        UBBAResult res;
        for (size_t it = 0; it < iters; ++it) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            res = solve_ubba_ladder_fast(blocks, total_w * 0.70, 0.10, true, 1.85f);
            const auto t1 = std::chrono::high_resolution_clock::now();
            lats.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }

        const auto stats = compute_stats(lats);
        const double throughput_m = (static_cast<double>(n) / stats.mean_us);

        std::cout << std::left << std::setw(12) << n
                  << std::setw(16) << std::fixed << std::setprecision(2) << stats.mean_us
                  << std::setw(16) << std::fixed << std::setprecision(4) << (stats.mean_us / 1000.0)
                  << std::setw(16) << res.num_blocks_selected
                  << std::setw(16) << std::fixed << std::setprecision(2) << throughput_m << "\n";
    }
}

// -----------------------------------------------------------------------------
// Benchmark 3: IFR LSE Engine & Vectorized Dot Products
// -----------------------------------------------------------------------------
void bench_ifr_lse(size_t num_candidates = 20000, size_t head_dim = 64) {
    print_banner("3. IFR LSE Engine & ARM Neon Vectorized Retrieval");

    LSEEngine lse_engine(head_dim);
    std::mt19937 rng(777);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);

    std::vector<float> query(head_dim);
    for (size_t d = 0; d < head_dim; ++d) query[d] = norm_dist(rng);

    std::vector<float> candidates(num_candidates * head_dim);
    for (size_t i = 0; i < candidates.size(); ++i) candidates[i] = norm_dist(rng);

    std::vector<float> logits(num_candidates);

    // 1. Logit dot-product throughput
    const size_t iters = 50;
    std::vector<double> logit_lats;
    logit_lats.reserve(iters);

    for (size_t it = 0; it < iters; ++it) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        lse_engine.compute_logits(query.data(), candidates.data(), num_candidates, logits.data());
        const auto t1 = std::chrono::high_resolution_clock::now();
        logit_lats.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    const auto logit_stats = compute_stats(logit_lats);
    const double logit_mops = (static_cast<double>(num_candidates) / logit_stats.mean_us);

    std::cout << "Candidates: " << num_candidates << " | Head Dim: " << head_dim << "\n";
    std::cout << ">>> Vectorized Logit Dot-Product:\n";
    std::cout << "    Mean Latency: " << logit_stats.mean_us << " us (" << (logit_stats.mean_us / 1000.0) << " ms)\n";
    std::cout << "    Throughput:   " << std::fixed << std::setprecision(2) << logit_mops << " Million dot-products/sec\n\n";

    // 2. Numerically stable LSE
    std::vector<double> lse_lats;
    lse_lats.reserve(iters);
    LSEResult lse_res;

    for (size_t it = 0; it < iters; ++it) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        lse_res = lse_engine.compute_lse(logits.data(), num_candidates);
        const auto t1 = std::chrono::high_resolution_clock::now();
        lse_lats.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    const auto lse_stats = compute_stats(lse_lats);
    std::cout << ">>> Numerically Stable Log-Sum-Exp Reduction:\n";
    std::cout << "    LSE: " << lse_res.lse << " | Z: " << lse_res.z_normalizer << "\n";
    std::cout << "    Mean Latency: " << lse_stats.mean_us << " us\n\n";

    // 3. Top-k unnormalized ranking & discarded attention mass
    const size_t top_k = 64;
    std::vector<size_t> top_indices;
    lse_engine.rank_unnormalized(logits.data(), num_candidates, top_k, top_indices);

    const float discarded_mass = lse_engine.compute_discarded_mass(
        logits.data(), num_candidates, top_indices.data(), top_indices.size(), lse_res.lse
    );

    std::cout << ">>> Top-" << top_k << " Unnormalized Ranking & Discarded Mass:\n";
    std::cout << "    Retained Candidates: " << top_indices.size() << "\n";
    std::cout << "    Discarded Attention Mass (rho): " << std::fixed << std::setprecision(4) << discarded_mass << "\n";
    std::cout << "    Fidelity Cliff (< 0.365): " << (discarded_mass <= 0.365f ? "MAINTAINED" : "EXCEEDED") << "\n";
}

// -----------------------------------------------------------------------------
// Benchmark 4: IFR Anti-Collapse Dispersion & Doublet Splitting
// -----------------------------------------------------------------------------
void bench_ifr_doublet_split(size_t num_blocks = 2000, size_t head_dim = 64) {
    print_banner("4. IFR Anti-Collapse Dispersion & Doublet Splitting");

    const size_t block_size = 32;
    AntiCollapseDispersionTracker tracker(0.85f, head_dim);

    std::mt19937 rng(999);
    std::normal_distribution<float> bg_dist(0.0f, 0.2f);
    std::uniform_real_distribution<float> burst_dist(2.0f, 4.0f);

    // Create synthetic blocks: 20% needles, 80% background
    std::vector<std::vector<float>> block_keys(num_blocks, std::vector<float>(block_size * head_dim));
    size_t expected_splits = 0;

    for (size_t b = 0; b < num_blocks; ++b) {
        const bool has_needle = (b % 5 == 0);
        if (has_needle) expected_splits++;

        for (size_t t = 0; t < block_size; ++t) {
            float* token_ptr = block_keys[b].data() + (t * head_dim);
            for (size_t d = 0; d < head_dim; ++d) {
                token_ptr[d] = bg_dist(rng);
            }
        }

        if (has_needle) {
            // Inject needle at token 7
            float* needle_ptr = block_keys[b].data() + (7 * head_dim);
            for (size_t d = 0; d < head_dim; ++d) {
                needle_ptr[d] += burst_dist(rng);
            }
        }
    }

    std::cout << "Evaluating " << num_blocks << " blocks (" << (num_blocks * block_size) 
              << " tokens), Threshold = 0.85\n";

    const auto t0 = std::chrono::high_resolution_clock::now();
    size_t actual_splits = 0;
    for (size_t b = 0; b < num_blocks; ++b) {
        auto res = tracker.inspect_and_split(block_keys[b].data(), block_size);
        if (res.is_split) {
            actual_splits++;
        }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double us_per_block = total_us / num_blocks;

    std::cout << ">>> Anti-Collapse Dispersion Results:\n";
    std::cout << "    Detected Splits: " << actual_splits << " (Expected ~" << expected_splits << ")\n";
    std::cout << "    Total Latency:   " << (total_us / 1000.0) << " ms\n";
    std::cout << "    Per-Block Time:  " << us_per_block << " us (" 
              << (num_blocks / (total_us / 1000.0)) << " blocks/ms)\n";
}

// -----------------------------------------------------------------------------
// Benchmark 5: CASA Zero-Copy Paged Block Table & Big-Tile Coalescing
// -----------------------------------------------------------------------------
void bench_casa_paged_table(size_t num_pages = 512, size_t head_dim = 64) {
    print_banner("5. CASA Zero-Copy Paged Block Table & Tensor Core Tile Evaluation");

    const size_t block_size = 32;
    CASAPagedBlockTable casa_table(block_size, head_dim, 4096);

    std::mt19937 rng(1234);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);

    std::vector<int32_t> page_table;
    page_table.reserve(num_pages);

    std::vector<float> k_unrot(block_size * head_dim);
    std::vector<float> v_data(block_size * head_dim);
    std::vector<int32_t> tokens(block_size);

    // Register pages into CASA store
    for (size_t p = 0; p < num_pages; ++p) {
        for (size_t t = 0; t < block_size; ++t) {
            tokens[t] = static_cast<int32_t>(p * block_size + t);
            for (size_t d = 0; d < head_dim; ++d) {
                k_unrot[t * head_dim + d] = norm_dist(rng);
                v_data[t * head_dim + d] = norm_dist(rng);
            }
        }
        const uint64_t hash = compute_prefix_hash_u64(tokens.data(), block_size);
        const int32_t phys_page = casa_table.allocate_and_store_page(
            tokens.data(), k_unrot.data(), v_data.data(), p * block_size, hash
        );
        page_table.push_back(phys_page);
    }

    std::cout << "Allocated and mapped " << page_table.size() << " physical pages ("
              << (page_table.size() * block_size) << " tokens).\n";

    // Zero-copy lookup speed
    const size_t lookup_iters = 1000000;
    const auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t checksum = 0;
    for (size_t i = 0; i < lookup_iters; ++i) {
        const size_t token_idx = (i * 17) % (num_pages * block_size);
        const float* ptr = casa_table.get_key_token_ptr(token_idx, page_table);
        checksum += reinterpret_cast<uintptr_t>(ptr);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double lookup_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double lookups_per_sec = (static_cast<double>(lookup_iters) / lookup_us) * 1e6;

    std::cout << ">>> Zero-Copy Page Table Address Translation:\n";
    std::cout << "    Lookups:  " << lookup_iters << " in " << (lookup_us / 1000.0) << " ms\n";
    std::cout << "    Speed:    " << std::fixed << std::setprecision(2) << (lookups_per_sec / 1e6) 
              << " Million zero-copy translations/sec (checksum: " << (checksum != 0 ? "valid" : "null") << ")\n\n";

    // PagedAttention GEMM Tile Execution
    std::vector<float> query_rot(head_dim);
    for (size_t d = 0; d < head_dim; ++d) query_rot[d] = norm_dist(rng);

    std::vector<float> out_scores;
    std::vector<float> out_context;

    const size_t paged_iters = 50;
    std::vector<double> paged_lats;
    paged_lats.reserve(paged_iters);

    for (size_t it = 0; it < paged_iters; ++it) {
        const auto tp0 = std::chrono::high_resolution_clock::now();
        casa_table.compute_paged_attention_tile(
            query_rot.data(), page_table, nullptr, out_scores, out_context
        );
        const auto tp1 = std::chrono::high_resolution_clock::now();
        paged_lats.push_back(std::chrono::duration<double, std::micro>(tp1 - tp0).count());
    }

    const auto paged_stats = compute_stats(paged_lats);
    const size_t total_tokens = num_pages * block_size;
    const double total_bytes = total_tokens * head_dim * sizeof(float) * 2; // K and V read
    const double effective_bandwidth_gb = (total_bytes / (paged_stats.mean_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);

    std::cout << ">>> PagedAttention Zero-Copy GEMM Tile Performance:\n";
    std::cout << "    Active Tokens:       " << total_tokens << "\n";
    std::cout << "    Mean Tile Latency:   " << paged_stats.mean_us << " us (" << (paged_stats.mean_us / 1000.0) << " ms)\n";
    std::cout << "    Effective Bandwidth: " << effective_bandwidth_gb << " GB/s\n\n";

    // Big-Tile Coalescing for GPUDirect Storage
    const auto super_tiles = casa_table.coalesce_big_tiles(page_table, 16);
    std::cout << ">>> Big-Tile Coalescing for GPUDirect Storage (GDS / cuFile):\n";
    std::cout << "    Formed " << super_tiles.size() << " Super-Tiles from " << page_table.size() << " logical blocks.\n";
    std::cout << "    Tile size: " << (super_tiles[0].size_bytes / 1024) << " KB (>= 64 KB saturation requirement met: "
              << (super_tiles[0].size_bytes >= 65536 ? "YES" : "NO") << ")\n";
}

// -----------------------------------------------------------------------------
// Verification: Bit-Level Compatibility & Invariant Checks
// -----------------------------------------------------------------------------
void verify_invariants() {
    print_banner("6. Bit-Level Mathematical Verification & Invariant Checks");

    // 1. Check RoPE algebraic equivalence: (R_m q)^T (R_n k) == q^T R_{m-n} k
    {
        const size_t dim = 64;
        std::vector<float> q(dim), k(dim);
        for (size_t d = 0; d < dim; ++d) {
            q[d] = 0.5f + static_cast<float>(d) * 0.02f;
            k[d] = 1.0f - static_cast<float>(d) * 0.01f;
        }

        const int64_t m = 42;
        const int64_t n = 15;

        std::vector<float> q_rot(dim), k_rot(dim), q_remap(dim);
        apply_rope_c(q.data(), q_rot.data(), dim, m);
        apply_rope_c(k.data(), k_rot.data(), dim, n);
        apply_rope_c(q.data(), q_remap.data(), dim, m - n);

        float dot_direct = 0.0f;
        float dot_remap = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            dot_direct += q_rot[d] * k_rot[d];
            dot_remap += q_remap[d] * k[d];
        }

        const float diff = std::abs(dot_direct - dot_remap);
        std::cout << ">>> RoPE Algebraic Invariance: (R_m q)^T (R_n k) == q^T R_{m-n} k:\n";
        std::cout << "    Direct Dot:  " << std::setprecision(8) << dot_direct << "\n";
        std::cout << "    Remap Dot:   " << std::setprecision(8) << dot_remap << "\n";
        std::cout << "    Difference:  " << diff << " (Machine Epsilon < 1e-6: "
                  << (diff < 1e-6f ? "PASS" : "FAIL") << ")\n\n";
        assert(diff < 1e-5f);
    }

    // 2. Check UBBA Softmax Tier Bias: b_t = -(rho_t * s)^2 / 2
    {
        TierProfile tp("INT2", 256, 0.343f, TierType::INT2);
        const float s = 1.85f;
        const float expected_sigma = 0.343f * 1.85f;
        const float expected_bias = -(expected_sigma * expected_sigma) * 0.5f;
        const float actual_bias = tp.get_tier_bias(s);
        const float bias_diff = std::abs(actual_bias - expected_bias);

        std::cout << ">>> UBBA Softmax Tier Bias: b_t = -(rho_t * s)^2 / 2:\n";
        std::cout << "    Expected:   " << expected_bias << "\n";
        std::cout << "    Computed:   " << actual_bias << "\n";
        std::cout << "    Difference: " << bias_diff << " (Match: " << (bias_diff < 1e-6f ? "PASS" : "FAIL") << ")\n\n";
        assert(bias_diff < 1e-6f);
    }

    // 3. Check LADDER Fidelity Cliff Clamping: rho_floor > 0.365 clamped to 0.365
    {
        std::vector<FastBlockCandidate> blocks;
        blocks.push_back(create_fast_ladder_candidate(0, 1.0f, 0.0f, 1024));

        auto res_clamped = solve_ubba_ladder_fast(blocks, 1.0, 0.50, true, 1.85f);
        std::cout << ">>> UBBA Fidelity Cliff Clamping:\n";
        std::cout << "    Configured rho_floor: 0.50 | Clamped rho_floor: " << res_clamped.rho_floor << "\n";
        std::cout << "    Selected Tier: " << res_clamped.allocations[0].tier_name 
                  << " (Expected INT2, MERGED blocked: " 
                  << (res_clamped.allocations[0].tier_name == "INT2" ? "PASS" : "FAIL") << ")\n";
        assert(std::abs(res_clamped.rho_floor - LADDER_FIDELITY_CLIFF) < 1e-9);
        assert(res_clamped.allocations[0].tier_name == "INT2");
    }

    std::cout << "\n>>> ALL BIT-LEVEL INVARIANTS AND SPECIFICATIONS VERIFIED SUCCESSFULLY [OK]\n";
}

int main(int argc, char* argv[]) {
    size_t num_blocks = 10000;
    size_t iterations = 50;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--num-blocks" && i + 1 < argc) {
            num_blocks = std::stoul(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoul(argv[++i]);
        }
    }

    std::cout << "===========================================================================\n";
    std::cout << "   KVMEM STRATA FUSION - TRACK B C++ HIGH PERFORMANCE ENGINE BENCHMARK     \n";
    std::cout << "===========================================================================\n";

    verify_invariants();
    bench_ubba_solver(num_blocks, iterations);
    bench_ubba_sweep();
    bench_ifr_lse(20000, 64);
    bench_ifr_doublet_split(2000, 64);
    bench_casa_paged_table(512, 64);

    print_banner("Benchmark Run Complete");
    return 0;
}
