#ifndef KVMEM_C_API_H
#define KVMEM_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Solve UBBA Knapsack with LADDER fidelity gate.
 * 
 * @param block_ids Array of block identifiers of length n
 * @param fidelities Array of attention/coverage weights (w_i >= 0) of length n
 * @param bytes_arr Array of byte costs (C_i > 0) of length n
 * @param n Number of candidate items
 * @param target_cov Target cumulative coverage demand
 * @param rho_floor Hard fidelity floor constraint
 * @param out_selected_ids Output array for selected block IDs (caller-allocated, capacity >= n)
 * @param out_selected_count Output pointer receiving the number of selected blocks
 * @return 0 on success, non-zero on error
 */
int kvmem_c_solve_ubba_ladder(
    const uint32_t* block_ids,
    const float* fidelities,
    const float* bytes_arr,
    uint32_t n,
    double target_cov,
    double rho_floor,
    uint32_t* out_selected_ids,
    uint32_t* out_selected_count
);

/**
 * Compute De-RoPE mean centroid and dispersion metrics for a key block.
 * 
 * @param k_block Pointer to row-major keys of shape [block_size, head_dim]
 * @param block_size Number of tokens in block (e.g., 32)
 * @param head_dim Key vector dimension (e.g., 64, 128)
 * @param out_mean Output buffer of size head_dim for the mean vector
 * @param out_dir_disp Output pointer for direct Euclidean max dispersion: max_i ||k_i - mean_k||
 * @param out_norm_disp Output pointer for directional max dispersion in cosine/L2-normalized space
 * @return 0 on success, non-zero on error
 */
int kvmem_c_compute_deroped_mean(
    const float* k_block,
    size_t block_size,
    size_t head_dim,
    float* out_mean,
    float* out_dir_disp,
    float* out_norm_disp
);

/**
 * Anti-collapse dispersion check and doublet centroid splitting.
 * 
 * @param k_block Pointer to row-major keys of shape [block_size, head_dim]
 * @param block_size Number of tokens in block
 * @param head_dim Key vector dimension
 * @param dispersion_threshold Threshold theta above which splitting is triggered
 * @param out_should_split Output pointer set to 1 if split occurred, 0 otherwise
 * @param out_c1 Output buffer of size head_dim (needle vector if split, mean if not split)
 * @param out_c2 Output buffer of size head_dim (residual mean if split, zeroes if not split)
 * @param out_dispersion Output pointer for measured dispersion metric
 * @return 0 on success, non-zero on error
 */
int kvmem_c_inspect_and_split(
    const float* k_block,
    size_t block_size,
    size_t head_dim,
    float dispersion_threshold,
    int* out_should_split,
    float* out_c1,
    float* out_c2,
    float* out_dispersion
);

/**
 * Decoupled sub-centroid cosine similarity ranking with top-K selection.
 * 
 * @param query Pointer to query vector of length dim
 * @param centroids Pointer to centroids array of shape [num_centroids, dim]
 * @param num_centroids Number of candidate centroids
 * @param dim Vector dimension
 * @param top_k Number of top candidates to return
 * @param out_indices Output buffer of size min(top_k, num_centroids) for ranked indices
 * @param out_scores Output buffer of size min(top_k, num_centroids) for cosine similarity scores
 * @return 0 on success, non-zero on error
 */
int kvmem_c_rank_subcentroids_cosine(
    const float* query,
    const float* centroids,
    size_t num_centroids,
    size_t dim,
    size_t top_k,
    size_t* out_indices,
    float* out_scores
);

/**
 * ARM Neon SIMD dot product of two float vectors.
 * 
 * @param a Pointer to first vector of length dim
 * @param b Pointer to second vector of length dim
 * @param dim Vector dimension
 * @return Dot product a . b
 */
float kvmem_c_neon_dot(
    const float* a,
    const float* b,
    size_t dim
);

/**
 * Zero-copy CASA paged block table address translation.
 * 
 * @param page_table Pointer to page table array (logical_page -> phys_page)
 * @param num_pages Number of entries in page table
 * @param logical_token_idx Logical token index
 * @param block_size Number of tokens per block
 * @param out_phys_page Output pointer for translated physical page ID
 * @param out_offset_in_page Output pointer for token offset within physical page
 * @return 0 on success, -1 if logical token is out of range
 */
int kvmem_c_paged_table_translate(
    const int32_t* page_table,
    size_t num_pages,
    size_t logical_token_idx,
    size_t block_size,
    int32_t* out_phys_page,
    size_t* out_offset_in_page
);

/**
 * Gather token vectors from physical memory pool via page table translation.
 * 
 * @param physical_pool Pointer to contiguous physical pool [max_pages * block_size * head_dim]
 * @param page_table Pointer to page table array
 * @param num_pages Number of pages in page table
 * @param token_indices Pointer to array of logical token indices
 * @param num_tokens Number of tokens to gather
 * @param block_size Number of tokens per block
 * @param head_dim Token vector dimension
 * @param out_gathered Output buffer of size [num_tokens * head_dim]
 * @return 0 on success, non-zero on error
 */
int kvmem_c_gather_paged_tokens(
    const float* physical_pool,
    const int32_t* page_table,
    size_t num_pages,
    const size_t* token_indices,
    size_t num_tokens,
    size_t block_size,
    size_t head_dim,
    float* out_gathered
);

#ifdef __cplusplus
}
#endif

#endif // KVMEM_C_API_H
