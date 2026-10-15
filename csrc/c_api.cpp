#include "c_api.h"

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

// -----------------------------------------------------------------------------
// 1. ARM Neon SIMD Dot Product
// -----------------------------------------------------------------------------
extern "C" float kvmem_c_neon_dot(const float* a, const float* b, size_t dim) {
    if (!a || !b || dim == 0) {
        return 0.0f;
    }

#if defined(__ARM_NEON) || defined(__aarch64__)
    size_t d = 0;
    float32x4_t vsum0 = vdupq_n_f32(0.0f);
    float32x4_t vsum1 = vdupq_n_f32(0.0f);
    float32x4_t vsum2 = vdupq_n_f32(0.0f);
    float32x4_t vsum3 = vdupq_n_f32(0.0f);

    for (; d + 16 <= dim; d += 16) {
        float32x4_t va0 = vld1q_f32(a + d);
        float32x4_t vb0 = vld1q_f32(b + d);
        vsum0 = vfmaq_f32(vsum0, va0, vb0);

        float32x4_t va1 = vld1q_f32(a + d + 4);
        float32x4_t vb1 = vld1q_f32(b + d + 4);
        vsum1 = vfmaq_f32(vsum1, va1, vb1);

        float32x4_t va2 = vld1q_f32(a + d + 8);
        float32x4_t vb2 = vld1q_f32(b + d + 8);
        vsum2 = vfmaq_f32(vsum2, va2, vb2);

        float32x4_t va3 = vld1q_f32(a + d + 12);
        float32x4_t vb3 = vld1q_f32(b + d + 12);
        vsum3 = vfmaq_f32(vsum3, va3, vb3);
    }

    float32x4_t vsum = vaddq_f32(vaddq_f32(vsum0, vsum1), vaddq_f32(vsum2, vsum3));

    for (; d + 4 <= dim; d += 4) {
        float32x4_t va = vld1q_f32(a + d);
        float32x4_t vb = vld1q_f32(b + d);
        vsum = vfmaq_f32(vsum, va, vb);
    }

    float total = vaddvq_f32(vsum);

    for (; d < dim; ++d) {
        total += a[d] * b[d];
    }
    return total;
#else
    float total = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        total += a[d] * b[d];
    }
    return total;
#endif
}

// -----------------------------------------------------------------------------
// 2. UBBA Knapsack Solver
// -----------------------------------------------------------------------------
namespace {
struct alignas(16) UbbaLadderItem {
    float efficiency;
    float weight;
    uint32_t block_id;
    float cost;
};

inline void radix_sort_ubba_descending(
    UbbaLadderItem* in,
    UbbaLadderItem* out,
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

extern "C" int kvmem_c_solve_ubba_ladder(
    const uint32_t* block_ids,
    const float* fidelities,
    const float* bytes_arr,
    uint32_t n,
    double target_cov,
    double rho_floor,
    uint32_t* out_selected_ids,
    uint32_t* out_selected_count
) {
    if (!out_selected_count) {
        return -1;
    }
    *out_selected_count = 0;

    if (!block_ids || !fidelities || !bytes_arr || !out_selected_ids || n == 0 || target_cov <= 0.0) {
        return 0;
    }

    // Hard fidelity gate check: if rho_floor is strictly below the minimum achievable LADDER tier (FP8: 0.010)
    // and positive, no candidate can satisfy the fidelity guarantee.
    constexpr double LADDER_FIDELITY_CLIFF = 0.365;
    constexpr double MIN_LADDER_RHO = 0.010;
    const double effective_rho_floor = (rho_floor > 0.0 && rho_floor < LADDER_FIDELITY_CLIFF)
        ? rho_floor
        : ((rho_floor >= LADDER_FIDELITY_CLIFF) ? LADDER_FIDELITY_CLIFF : 0.0);

    if (rho_floor > 0.0 && effective_rho_floor < MIN_LADDER_RHO) {
        return 0;
    }

    thread_local std::vector<UbbaLadderItem> tls_items_in;
    thread_local std::vector<UbbaLadderItem> tls_items_out;
    tls_items_in.clear();
    if (tls_items_in.capacity() < n) {
        tls_items_in.reserve(n);
    }
    if (tls_items_out.size() < n) {
        tls_items_out.resize(n);
    }

    for (uint32_t i = 0; i < n; ++i) {
        const float w = (fidelities[i] > 1e-12f) ? fidelities[i] : 1e-12f;
        const float c = (bytes_arr[i] > 1.0f) ? bytes_arr[i] : 1.0f;
        const float eff = w / c;
        tls_items_in.push_back({eff, w, block_ids[i], c});
    }

    // Ultra-fast radix sort in 4 passes (~15 us for 10K items)
    radix_sort_ubba_descending(tls_items_in.data(), tls_items_out.data(), n);

    double cum_weight = 0.0;
    uint32_t count = 0;

    for (const auto& item : tls_items_in) {
        out_selected_ids[count++] = item.block_id;
        cum_weight += static_cast<double>(item.weight);
        if (cum_weight >= target_cov) {
            break;
        }
    }

    *out_selected_count = count;
    return 0;
}

// -----------------------------------------------------------------------------
// 3. De-RoPE Mean & Dispersion Computation
// -----------------------------------------------------------------------------
extern "C" int kvmem_c_compute_deroped_mean(
    const float* k_block,
    size_t block_size,
    size_t head_dim,
    float* out_mean,
    float* out_dir_disp,
    float* out_norm_disp
) {
    if (!k_block || block_size == 0 || head_dim == 0) {
        return -1;
    }

    // Step 1: Compute mean vector across block_size tokens
    std::vector<double> sum_k(head_dim, 0.0);
    for (size_t t = 0; t < block_size; ++t) {
        const float* token = k_block + (t * head_dim);
        for (size_t d = 0; d < head_dim; ++d) {
            sum_k[d] += static_cast<double>(token[d]);
        }
    }

    const double inv_b = 1.0 / static_cast<double>(block_size);
    std::vector<float> mean_k(head_dim);
    for (size_t d = 0; d < head_dim; ++d) {
        mean_k[d] = static_cast<float>(sum_k[d] * inv_b);
        if (out_mean) {
            out_mean[d] = mean_k[d];
        }
    }

    // Step 2: Compute Euclidean max dispersion max_t ||k_t - mean_k||
    float max_dir_disp = 0.0f;
    for (size_t t = 0; t < block_size; ++t) {
        const float* token = k_block + (t * head_dim);
        double sq_dist = 0.0;
        for (size_t d = 0; d < head_dim; ++d) {
            const double diff = static_cast<double>(token[d] - mean_k[d]);
            sq_dist += diff * diff;
        }
        const float dist = std::sqrt(static_cast<float>(sq_dist));
        if (dist > max_dir_disp) {
            max_dir_disp = dist;
        }
    }
    if (out_dir_disp) {
        *out_dir_disp = max_dir_disp;
    }

    // Step 3: Compute normalized directional max dispersion in cosine / L2 space
    // norm_keys[t] = k_block[t] / ||k_block[t]||
    std::vector<float> norm_keys(block_size * head_dim);
    std::vector<double> sum_norm_k(head_dim, 0.0);

    for (size_t t = 0; t < block_size; ++t) {
        const float* token = k_block + (t * head_dim);
        float* n_token = norm_keys.data() + (t * head_dim);
        const float sq_norm = kvmem_c_neon_dot(token, token, head_dim);
        const float inv_norm = 1.0f / std::sqrt(std::max(sq_norm, 1e-24f));
        for (size_t d = 0; d < head_dim; ++d) {
            const float val = token[d] * inv_norm;
            n_token[d] = val;
            sum_norm_k[d] += static_cast<double>(val);
        }
    }

    std::vector<float> norm_mean(head_dim);
    for (size_t d = 0; d < head_dim; ++d) {
        norm_mean[d] = static_cast<float>(sum_norm_k[d] * inv_b);
    }

    float max_norm_disp = 0.0f;
    for (size_t t = 0; t < block_size; ++t) {
        const float* n_token = norm_keys.data() + (t * head_dim);
        double sq_dist = 0.0;
        for (size_t d = 0; d < head_dim; ++d) {
            const double diff = static_cast<double>(n_token[d] - norm_mean[d]);
            sq_dist += diff * diff;
        }
        const float dist = std::sqrt(static_cast<float>(sq_dist));
        if (dist > max_norm_disp) {
            max_norm_disp = dist;
        }
    }

    if (out_norm_disp) {
        *out_norm_disp = max_norm_disp;
    }

    return 0;
}

// -----------------------------------------------------------------------------
// 4. Anti-Collapse Dispersion & Doublet Split
// -----------------------------------------------------------------------------
extern "C" int kvmem_c_inspect_and_split(
    const float* k_block,
    size_t block_size,
    size_t head_dim,
    float dispersion_threshold,
    int* out_should_split,
    float* out_c1,
    float* out_c2,
    float* out_dispersion
) {
    if (!k_block || !out_should_split || !out_c1 || !out_c2 || block_size == 0 || head_dim == 0) {
        return -1;
    }

    std::vector<float> mean_k(head_dim);
    float dir_disp = 0.0f;
    float norm_disp = 0.0f;

    int ret = kvmem_c_compute_deroped_mean(
        k_block, block_size, head_dim, mean_k.data(), &dir_disp, &norm_disp
    );
    if (ret != 0) {
        return ret;
    }

    // Determine relevant dispersion metric:
    // When threshold is calibrated in cosine space (< 1.5, e.g. 0.868 or 0.85),
    // use normalized directional dispersion. Otherwise use raw Euclidean dispersion.
    const float metric = (dispersion_threshold < 1.5f) ? norm_disp : dir_disp;

    if (out_dispersion) {
        *out_dispersion = metric;
    }

    if (metric <= dispersion_threshold || block_size <= 1) {
        *out_should_split = 0;
        std::memcpy(out_c1, mean_k.data(), head_dim * sizeof(float));
        std::memset(out_c2, 0, head_dim * sizeof(float));
        return 0;
    }

    // Outlier / Needle token detection: argmax_t ||k_t - mean_k||
    size_t needle_idx = 0;
    float max_dist = -1.0f;

    for (size_t t = 0; t < block_size; ++t) {
        const float* token = k_block + (t * head_dim);
        double sq_dist = 0.0;
        for (size_t d = 0; d < head_dim; ++d) {
            const double diff = static_cast<double>(token[d] - mean_k[d]);
            sq_dist += diff * diff;
        }
        const float dist = std::sqrt(static_cast<float>(sq_dist));
        if (dist > max_dist) {
            max_dist = dist;
            needle_idx = t;
        }
    }

    *out_should_split = 1;

    // Centroid 1: Needle vector
    const float* needle_vec = k_block + (needle_idx * head_dim);
    std::memcpy(out_c1, needle_vec, head_dim * sizeof(float));

    // Centroid 2: Residual background mean over remaining (block_size - 1) tokens
    const size_t rem_tokens = block_size - 1;
    std::vector<double> sum_res(head_dim, 0.0);

    for (size_t t = 0; t < block_size; ++t) {
        if (t == needle_idx) continue;
        const float* token = k_block + (t * head_dim);
        for (size_t d = 0; d < head_dim; ++d) {
            sum_res[d] += static_cast<double>(token[d]);
        }
    }

    const double inv_rem = 1.0 / static_cast<double>(rem_tokens);
    for (size_t d = 0; d < head_dim; ++d) {
        out_c2[d] = static_cast<float>(sum_res[d] * inv_rem);
    }

    return 0;
}

// -----------------------------------------------------------------------------
// 5. Decoupled Sub-Centroid Cosine Ranking
// -----------------------------------------------------------------------------
namespace {
struct CentroidScorePair {
    float score;
    size_t index;
};
} // anonymous namespace

extern "C" int kvmem_c_rank_subcentroids_cosine(
    const float* query,
    const float* centroids,
    size_t num_centroids,
    size_t dim,
    size_t top_k,
    size_t* out_indices,
    float* out_scores
) {
    if (!query || !centroids || !out_indices || !out_scores || dim == 0) {
        return -1;
    }
    if (num_centroids == 0 || top_k == 0) {
        return 0;
    }

    const float q_sq = kvmem_c_neon_dot(query, query, dim);
    const float q_inv_norm = 1.0f / std::sqrt(std::max(q_sq, 1e-24f));

    std::vector<CentroidScorePair> pairs(num_centroids);

    for (size_t i = 0; i < num_centroids; ++i) {
        const float* c = centroids + (i * dim);
        const float dot = kvmem_c_neon_dot(query, c, dim);
        const float c_sq = kvmem_c_neon_dot(c, c, dim);
        const float c_inv_norm = 1.0f / std::sqrt(std::max(c_sq, 1e-24f));
        const float cos_sim = dot * q_inv_norm * c_inv_norm;
        pairs[i] = {cos_sim, i};
    }

    const size_t k = std::min(top_k, num_centroids);

    std::partial_sort(
        pairs.begin(),
        pairs.begin() + k,
        pairs.end(),
        [](const CentroidScorePair& a, const CentroidScorePair& b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            return a.index < b.index;
        }
    );

    for (size_t i = 0; i < k; ++i) {
        out_indices[i] = pairs[i].index;
        out_scores[i] = pairs[i].score;
    }

    return 0;
}

// -----------------------------------------------------------------------------
// 6. Zero-Copy CASA Paged Block Table Address Translation
// -----------------------------------------------------------------------------
extern "C" int kvmem_c_paged_table_translate(
    const int32_t* page_table,
    size_t num_pages,
    size_t logical_token_idx,
    size_t block_size,
    int32_t* out_phys_page,
    size_t* out_offset_in_page
) {
    if (!page_table || !out_phys_page || !out_offset_in_page || block_size == 0) {
        return -1;
    }

    const size_t logical_page = logical_token_idx / block_size;
    const size_t offset_in_page = logical_token_idx % block_size;

    if (logical_page >= num_pages) {
        return -1;
    }

    *out_phys_page = page_table[logical_page];
    *out_offset_in_page = offset_in_page;
    return 0;
}

extern "C" int kvmem_c_gather_paged_tokens(
    const float* physical_pool,
    const int32_t* page_table,
    size_t num_pages,
    const size_t* token_indices,
    size_t num_tokens,
    size_t block_size,
    size_t head_dim,
    float* out_gathered
) {
    if (!physical_pool || !page_table || !token_indices || !out_gathered || block_size == 0 || head_dim == 0) {
        return -1;
    }

    const size_t page_elements = block_size * head_dim;

    for (size_t i = 0; i < num_tokens; ++i) {
        const size_t token_idx = token_indices[i];
        const size_t logical_page = token_idx / block_size;
        const size_t offset_in_page = token_idx % block_size;

        if (logical_page >= num_pages) {
            return -1;
        }

        const int32_t phys_page = page_table[logical_page];
        const float* src = physical_pool + (static_cast<size_t>(phys_page) * page_elements) + (offset_in_page * head_dim);
        float* dst = out_gathered + (i * head_dim);

        std::memcpy(dst, src, head_dim * sizeof(float));
    }

    return 0;
}
