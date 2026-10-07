#include "ifr_lse.hpp"

#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstring>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace kvmem {

namespace {

inline float dot_product_simd(const float* a, const float* b, size_t dim) {
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

inline float find_max_simd(const float* data, size_t n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    size_t i = 0;
    float32x4_t vmax = vdupq_n_f32(-1e30f);

    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        vmax = vmaxq_f32(vmax, v);
    }

    float max_val = vmaxvq_f32(vmax);
    for (; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
#else
    float max_val = -1e30f;
    for (size_t i = 0; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
#endif
}

} // anonymous namespace

LSEEngine::LSEEngine(size_t head_dim)
    : head_dim_(head_dim),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim))) {}

void LSEEngine::compute_logits(
    const float* query,
    const float* candidates,
    size_t num_candidates,
    float* out_logits
) const {
    for (size_t j = 0; j < num_candidates; ++j) {
        const float* cand = candidates + (j * head_dim_);
        const float dot = dot_product_simd(query, cand, head_dim_);
        out_logits[j] = dot * scale_;
    }
}

LSEResult LSEEngine::compute_lse(const float* logits, size_t n) const {
    if (n == 0) {
        return {0.0f, 0.0f};
    }

    const float max_s = find_max_simd(logits, n);
    double sum_exp = 0.0;

    for (size_t i = 0; i < n; ++i) {
        sum_exp += std::exp(static_cast<double>(logits[i] - max_s));
    }

    const float lse = static_cast<float>(static_cast<double>(max_s) + std::log(sum_exp));
    const float z = std::exp(lse);
    return {lse, z};
}

float LSEEngine::compute_discarded_mass(
    const float* full_logits,
    size_t n_full,
    const size_t* selected_indices,
    size_t n_selected,
    float full_lse
) const {
    if (n_selected == 0 || n_full == 0) {
        return 1.0f;
    }

    const float max_s = find_max_simd(full_logits, n_full);
    double selected_sum_exp = 0.0;

    for (size_t i = 0; i < n_selected; ++i) {
        const size_t idx = selected_indices[i];
        selected_sum_exp += std::exp(static_cast<double>(full_logits[idx] - max_s));
    }

    const double selected_lse = static_cast<double>(max_s) + std::log(std::max(selected_sum_exp, 1e-30));
    const double log_retained_ratio = selected_lse - static_cast<double>(full_lse);
    const double retained_mass = std::exp(std::min(0.0, log_retained_ratio));
    const float discarded_mass = static_cast<float>(std::max(0.0, 1.0 - retained_mass));

    return discarded_mass;
}

void LSEEngine::rank_unnormalized(
    const float* logits,
    size_t n,
    size_t top_k,
    std::vector<size_t>& out_indices
) const {
    out_indices.resize(n);
    std::iota(out_indices.begin(), out_indices.end(), 0);

    const size_t k = std::min(top_k, n);

    std::partial_sort(
        out_indices.begin(),
        out_indices.begin() + k,
        out_indices.end(),
        [logits](size_t a, size_t b) {
            return logits[a] > logits[b];
        }
    );

    out_indices.resize(k);
}

AntiCollapseDispersionTracker::AntiCollapseDispersionTracker(
    float dispersion_threshold,
    size_t head_dim
) : dispersion_threshold_(dispersion_threshold),
    head_dim_(head_dim) {}

void AntiCollapseDispersionTracker::compute_deroped_mean(
    const float* keys,
    size_t num_tokens,
    float* out_mean,
    float& out_max_dispersion,
    float& out_avg_dispersion
) const {
    if (num_tokens == 0) {
        out_max_dispersion = 0.0f;
        out_avg_dispersion = 0.0f;
        return;
    }

    // Step 1: Compute mean vector across tokens
    std::vector<double> sum_k(head_dim_, 0.0);
    for (size_t t = 0; t < num_tokens; ++t) {
        const float* k_token = keys + (t * head_dim_);
        for (size_t d = 0; d < head_dim_; ++d) {
            sum_k[d] += static_cast<double>(k_token[d]);
        }
    }

    const double inv_n = 1.0 / static_cast<double>(num_tokens);
    for (size_t d = 0; d < head_dim_; ++d) {
        out_mean[d] = static_cast<float>(sum_k[d] * inv_n);
    }

    // Step 2: Compute Euclidean distances from mean
    float max_dist = 0.0f;
    double sum_dist = 0.0;

    for (size_t t = 0; t < num_tokens; ++t) {
        const float* k_token = keys + (t * head_dim_);
        double sq_dist = 0.0;

#if defined(__ARM_NEON) || defined(__aarch64__)
        size_t d = 0;
        float32x4_t vsq = vdupq_n_f32(0.0f);
        for (; d + 4 <= head_dim_; d += 4) {
            float32x4_t vk = vld1q_f32(k_token + d);
            float32x4_t vm = vld1q_f32(out_mean + d);
            float32x4_t diff = vsubq_f32(vk, vm);
            vsq = vfmaq_f32(vsq, diff, diff);
        }
        sq_dist = static_cast<double>(vaddvq_f32(vsq));
        for (; d < head_dim_; ++d) {
            const double diff = static_cast<double>(k_token[d] - out_mean[d]);
            sq_dist += diff * diff;
        }
#else
        for (size_t d = 0; d < head_dim_; ++d) {
            const double diff = static_cast<double>(k_token[d] - out_mean[d]);
            sq_dist += diff * diff;
        }
#endif
        const float dist = std::sqrt(static_cast<float>(sq_dist));
        if (dist > max_dist) {
            max_dist = dist;
        }
        sum_dist += static_cast<double>(dist);
    }

    out_max_dispersion = max_dist;
    out_avg_dispersion = static_cast<float>(sum_dist * inv_n);
}

DoubletSplitResult AntiCollapseDispersionTracker::inspect_and_split(
    const float* keys,
    size_t num_tokens
) const {
    DoubletSplitResult res;
    res.mean_centroid.resize(head_dim_);

    compute_deroped_mean(
        keys,
        num_tokens,
        res.mean_centroid.data(),
        res.max_dispersion,
        res.avg_dispersion
    );

    if (res.max_dispersion <= dispersion_threshold_ || num_tokens <= 1) {
        res.is_split = false;
        res.needle_token_idx = 0;
        return res;
    }

    // Needle detection: token with maximum distance from mean
    res.is_split = true;
    size_t needle_idx = 0;
    float max_dist = -1.0f;

    for (size_t t = 0; t < num_tokens; ++t) {
        const float* k_token = keys + (t * head_dim_);
        double sq_dist = 0.0;
        for (size_t d = 0; d < head_dim_; ++d) {
            const double diff = static_cast<double>(k_token[d] - res.mean_centroid[d]);
            sq_dist += diff * diff;
        }
        const float dist = std::sqrt(static_cast<float>(sq_dist));
        if (dist > max_dist) {
            max_dist = dist;
            needle_idx = t;
        }
    }

    res.needle_token_idx = needle_idx;

    // Centroid 1: Needle vector
    res.needle_centroid.resize(head_dim_);
    const float* needle_vec = keys + (needle_idx * head_dim_);
    std::memcpy(res.needle_centroid.data(), needle_vec, head_dim_ * sizeof(float));

    // Centroid 2: Residual background mean over remaining (num_tokens - 1) tokens
    res.residual_centroid.resize(head_dim_, 0.0f);
    std::vector<double> sum_res(head_dim_, 0.0);
    const size_t rem_tokens = num_tokens - 1;

    for (size_t t = 0; t < num_tokens; ++t) {
        if (t == needle_idx) continue;
        const float* k_token = keys + (t * head_dim_);
        for (size_t d = 0; d < head_dim_; ++d) {
            sum_res[d] += static_cast<double>(k_token[d]);
        }
    }

    const double inv_rem = 1.0 / static_cast<double>(rem_tokens);
    for (size_t d = 0; d < head_dim_; ++d) {
        res.residual_centroid[d] = static_cast<float>(sum_res[d] * inv_rem);
    }

    return res;
}

} // namespace kvmem
