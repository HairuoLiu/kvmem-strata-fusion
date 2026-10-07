#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace kvmem {

// Forward RoPE rotation for 2D coordinate pairs
inline void apply_rope_c(
    const float* src,
    float* dst,
    size_t dim,
    int64_t pos,
    float base = 10000.0f
) {
    const size_t half_dim = dim / 2;
    for (size_t i = 0; i < half_dim; ++i) {
        const float theta = static_cast<float>(pos) * std::pow(base, -2.0f * static_cast<float>(i) / static_cast<float>(dim));
        const float cos_t = std::cos(theta);
        const float sin_t = std::sin(theta);

        const float x_even = src[2 * i];
        const float x_odd  = src[2 * i + 1];

        dst[2 * i]     = x_even * cos_t - x_odd * sin_t;
        dst[2 * i + 1] = x_even * sin_t + x_odd * cos_t;
    }
}

// 64-bit FNV-1a hash for causal prefix hash chains
inline uint64_t compute_prefix_hash_u64(
    const int32_t* tokens,
    size_t num_tokens,
    uint64_t parent_hash = 0xcbf29ce484222325ULL // FNV offset basis
) {
    uint64_t h = parent_hash;
    const uint64_t prime = 0x100000001b3ULL;

    for (size_t i = 0; i < num_tokens; ++i) {
        const uint32_t tok = static_cast<uint32_t>(tokens[i]);
        h ^= (tok & 0xFF);
        h *= prime;
        h ^= ((tok >> 8) & 0xFF);
        h *= prime;
        h ^= ((tok >> 16) & 0xFF);
        h *= prime;
        h ^= ((tok >> 24) & 0xFF);
        h *= prime;
    }
    return h;
}

// Storage Super-Tile representation for GDS / cuFile big-tile coalescing
struct StorageSuperTileDesc {
    uint32_t tile_id;
    std::vector<int32_t> physical_pages;
    size_t total_tokens;
    size_t size_bytes;
    size_t nvme_offset_bytes;
};

class CASAPagedBlockTable {
public:
    CASAPagedBlockTable(
        size_t block_size = 32,
        size_t head_dim = 64,
        size_t max_pages = 65536
    ) : block_size_(block_size),
        head_dim_(head_dim),
        max_pages_(max_pages),
        page_elements_(block_size * head_dim),
        scale_(1.0f / std::sqrt(static_cast<float>(head_dim))) {
        // Allocate contiguous physical pools for zero-copy access
        k_physical_pool_.resize(max_pages * page_elements_, 0.0f);
        v_physical_pool_.resize(max_pages * page_elements_, 0.0f);
        ref_counts_.resize(max_pages, 0);
        page_tiers_.resize(max_pages, 0);
        allocated_pages_ = 0;
    }

    // Zero-copy direct pointer to physical Key page
    inline const float* get_key_page_ptr(int32_t physical_page_id) const noexcept {
        return k_physical_pool_.data() + (static_cast<size_t>(physical_page_id) * page_elements_);
    }

    // Zero-copy direct pointer to physical Value page
    inline const float* get_val_page_ptr(int32_t physical_page_id) const noexcept {
        return v_physical_pool_.data() + (static_cast<size_t>(physical_page_id) * page_elements_);
    }

    // Zero-copy direct token pointer via page table translation
    inline const float* get_key_token_ptr(
        size_t logical_token_idx,
        const std::vector<int32_t>& page_table
    ) const {
        const size_t logical_page = logical_token_idx / block_size_;
        const size_t offset_in_page = logical_token_idx % block_size_;
        if (logical_page >= page_table.size()) {
            return nullptr;
        }
        const int32_t phys_page = page_table[logical_page];
        return get_key_page_ptr(phys_page) + (offset_in_page * head_dim_);
    }

    // Allocate and register a canonical immutable atom page
    // k_unrotated is rotated once at orig_pos_start (K-Freeze) and stored permanently
    int32_t allocate_and_store_page(
        const int32_t* tokens,
        const float* k_unrotated,
        const float* v,
        int64_t orig_pos_start,
        uint64_t prefix_hash,
        uint8_t tier_id = 0
    ) {
        (void)tokens;
        // Check prefix hash deduplication
        auto it = prefix_to_page_.find(prefix_hash);
        if (it != prefix_to_page_.end()) {
            const int32_t existing_page = it->second;
            ref_counts_[existing_page]++;
            return existing_page;
        }

        if (allocated_pages_ >= max_pages_) {
            throw std::runtime_error("CASA physical page pool exhausted");
        }

        const int32_t new_page = static_cast<int32_t>(allocated_pages_++);
        ref_counts_[new_page] = 1;
        page_tiers_[new_page] = tier_id;
        prefix_to_page_[prefix_hash] = new_page;

        float* k_dst = k_physical_pool_.data() + (new_page * page_elements_);
        float* v_dst = v_physical_pool_.data() + (new_page * page_elements_);

        // K-Freeze: Rotate Key vectors once to canonical position and store immutably
        for (size_t t = 0; t < block_size_; ++t) {
            apply_rope_c(
                k_unrotated + (t * head_dim_),
                k_dst + (t * head_dim_),
                head_dim_,
                orig_pos_start + static_cast<int64_t>(t)
            );
        }

        // Value copied verbatim
        std::memcpy(v_dst, v, page_elements_ * sizeof(float));

        return new_page;
    }

    // Compute PagedAttention GEMM tile using zero-copy page table lookups
    // query_rot: Query pre-rotated at sequence generation position
    void compute_paged_attention_tile(
        const float* query_rot,
        const std::vector<int32_t>& page_table,
        const float* tier_biases,  // optional per-page bias array
        std::vector<float>& out_scores,
        std::vector<float>& out_context
    ) const {
        const size_t num_pages = page_table.size();
        const size_t total_tokens = num_pages * block_size_;

        out_scores.resize(total_tokens);
        out_context.assign(head_dim_, 0.0f);

        // Step 1: Batched GEMM tile dot-products directly against physical pages
        float max_score = -1e30f;

        for (size_t p = 0; p < num_pages; ++p) {
            const int32_t phys_page = page_table[p];
            const float* k_page = get_key_page_ptr(phys_page);
            const float bias = tier_biases ? tier_biases[p] : 0.0f;
            float* score_dst = out_scores.data() + (p * block_size_);

            for (size_t t = 0; t < block_size_; ++t) {
                const float* k_vec = k_page + (t * head_dim_);

                // Direct dot product
                float dot = 0.0f;
#if defined(__ARM_NEON) || defined(__aarch64__)
                size_t d = 0;
                float32x4_t vsum = vdupq_n_f32(0.0f);
                for (; d + 4 <= head_dim_; d += 4) {
                    float32x4_t vq = vld1q_f32(query_rot + d);
                    float32x4_t vk = vld1q_f32(k_vec + d);
                    vsum = vfmaq_f32(vsum, vq, vk);
                }
                dot = vaddvq_f32(vsum);
                for (; d < head_dim_; ++d) {
                    dot += query_rot[d] * k_vec[d];
                }
#else
                for (size_t d = 0; d < head_dim_; ++d) {
                    dot += query_rot[d] * k_vec[d];
                }
#endif
                const float s = (dot * scale_) + bias;
                score_dst[t] = s;
                if (s > max_score) {
                    max_score = s;
                }
            }
        }

        // Step 2: Softmax normalization
        double sum_exp = 0.0;
        for (size_t i = 0; i < total_tokens; ++i) {
            const double e = std::exp(static_cast<double>(out_scores[i] - max_score));
            out_scores[i] = static_cast<float>(e);
            sum_exp += e;
        }

        const double inv_sum = 1.0 / (sum_exp > 0.0 ? sum_exp : 1.0);
        for (size_t i = 0; i < total_tokens; ++i) {
            out_scores[i] = static_cast<float>(out_scores[i] * inv_sum);
        }

        // Step 3: Context vector reduction: context = sum_i attn_weights[i] * V[i]
        for (size_t p = 0; p < num_pages; ++p) {
            const int32_t phys_page = page_table[p];
            const float* v_page = get_val_page_ptr(phys_page);
            const float* weights = out_scores.data() + (p * block_size_);

            for (size_t t = 0; t < block_size_; ++t) {
                const float w = weights[t];
                const float* v_vec = v_page + (t * head_dim_);
                for (size_t d = 0; d < head_dim_; ++d) {
                    out_context[d] += w * v_vec[d];
                }
            }
        }
    }

    // Coalesce fine-grained logical pages into Big-Tile macro chunks for GPUDirect Storage DMA
    std::vector<StorageSuperTileDesc> coalesce_big_tiles(
        const std::vector<int32_t>& page_table,
        size_t pages_per_tile = 16
    ) const {
        std::vector<StorageSuperTileDesc> tiles;
        const size_t bytes_per_token = head_dim_ * 2 * sizeof(float); // K and V
        const size_t bytes_per_block = block_size_ * bytes_per_token;

        for (size_t i = 0; i < page_table.size(); i += pages_per_tile) {
            const size_t chunk_len = std::min(pages_per_tile, page_table.size() - i);
            StorageSuperTileDesc tile;
            tile.tile_id = static_cast<uint32_t>(tiles.size());
            tile.physical_pages.assign(page_table.begin() + i, page_table.begin() + i + chunk_len);
            tile.total_tokens = chunk_len * block_size_;
            tile.size_bytes = chunk_len * bytes_per_block;
            tile.nvme_offset_bytes = tiles.size() * pages_per_tile * bytes_per_block;
            tiles.push_back(std::move(tile));
        }
        return tiles;
    }

    size_t block_size() const { return block_size_; }
    size_t head_dim() const { return head_dim_; }
    size_t allocated_pages() const { return allocated_pages_; }
    size_t max_pages() const { return max_pages_; }
    int32_t ref_count(int32_t page_id) const { return ref_counts_[page_id]; }

private:
    size_t block_size_;
    size_t head_dim_;
    size_t max_pages_;
    size_t page_elements_;
    float scale_;
    size_t allocated_pages_;

    // Continuous physical memory pools
    std::vector<float> k_physical_pool_;
    std::vector<float> v_physical_pool_;
    std::vector<int32_t> ref_counts_;
    std::vector<uint8_t> page_tiers_;

    // Prefix hash -> physical page ID table
    std::unordered_map<uint64_t, int32_t> prefix_to_page_;
};

} // namespace kvmem
