#pragma once

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <immintrin.h>

// 归一化目标总权重（< 2^31，留 0.8% 余量防 int32 溢出）
static constexpr float TARGET_TOTAL_F = 2.13e9f;

// ---------------------------------------------------------------------------
// 两段式管线（调用方负责段间 barrier；各段按 2048 元素块做 tid/nt 划分）：
//
//   第一部分 categorical_build_weights —— 实例化 int 权重 wi：
//     读 probs（可选 top-p 修正：p' = p>th ? p : (p==th ? p*comp : 0)，
//     th<=0 时无修正），乘 factor 截断，写 weights[i] = (int)p'。
//
//   第二部分 categorical_build_local_prefixes + categorical_add_local_offsets
//   —— 由 wi 计算前缀和：
//     前者把块内包含型局部前缀和写入 prefix_sum、块总权重写入 block_sums；
//     后者把全局块偏移加到 prefix_sum。
//
//   采样 categorical_total_weight / categorical_sample。
//
// 确定性约定：权重只依赖 probs/factor/th/comp 的逐元素值（与线程划分无关），
// 偏移按块序号定序累加，因此同输入下多线程与单线程结果逐位一致。
// 调用方需保证 probs 非负、factor 有限且总权重 > 0。
// ---------------------------------------------------------------------------

// ---- 第一部分：int 权重实例化（元素级运算，块间无依赖） ----
inline void categorical_build_weights(const float* probs, size_t n, int* weights,
                                      float factor, float th, float comp,
                                      int tid, int nt) {
    constexpr size_t block_size = 2048;
    const size_t num_blocks = (n + block_size - 1) / block_size;
    size_t b0 = (num_blocks * (size_t)tid) / (size_t)nt;
    size_t b1 = (num_blocks * (size_t)(tid + 1)) / (size_t)nt;

    const __m256 fv = _mm256_set1_ps(factor);
    const __m256 thv = _mm256_set1_ps(th);
    const __m256 cv = _mm256_set1_ps(comp);

    for (size_t b = b0; b < b1; ++b) {
        size_t start = b * block_size;
        size_t end = std::min(start + block_size, n);
        if (start >= end) continue;

        size_t i = start;
        for (; i + 8 <= end; i += 8) {
            __m256 p = _mm256_loadu_ps(probs + i);
            // 可选 top-p 修正
            __m256 gt = _mm256_cmp_ps(p, thv, _CMP_GT_OQ);
            __m256 eq = _mm256_cmp_ps(p, thv, _CMP_EQ_OQ);
            __m256 pp = _mm256_or_ps(_mm256_and_ps(p, gt),
                                     _mm256_mul_ps(_mm256_and_ps(p, eq), cv));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(weights + i),
                                _mm256_cvttps_epi32(_mm256_mul_ps(pp, fv)));
        }
        for (; i < end; ++i) {
            float p = probs[i];
            if (th > 0.0f) p = (p > th) ? p : ((p == th) ? (p * comp) : 0.0f);
            weights[i] = static_cast<int>(p * factor);
        }
    }
}

// ---- 第二部分（a）：由 wi 计算块内包含型局部前缀和 + 块总权重 ----
inline void categorical_build_local_prefixes(const int* weights, size_t n, int* prefix_sum,
                                             int* block_sums, int tid, int nt) {
    constexpr size_t block_size = 2048;
    const size_t num_blocks = (n + block_size - 1) / block_size;
    size_t b0 = (num_blocks * (size_t)tid) / (size_t)nt;
    size_t b1 = (num_blocks * (size_t)(tid + 1)) / (size_t)nt;

    const __m256i idx4 = _mm256_setr_epi32(7, 7, 7, 7, 0, 1, 2, 3);
    const __m256i idx2 = _mm256_setr_epi32(7, 7, 0, 1, 2, 3, 4, 5);
    const __m256i idx1 = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
    const __m256i zeroi = _mm256_setzero_si256();

    for (size_t b = b0; b < b1; ++b) {
        size_t start = b * block_size;
        size_t end = std::min(start + block_size, n);
        if (start >= end) { block_sums[b] = 0; continue; }

        int local_sum = 0;
        size_t i = start;
        // 主循环：8 个 int / 迭代
        for (; i + 8 <= end; i += 8) {
            __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + i));
            // Hillis-Steele int32 前缀和（8 lane，位移 4/2/1）
            __m256i s = _mm256_add_epi32(wi,
                         _mm256_blend_epi32(_mm256_permutevar8x32_epi32(wi, idx4), zeroi, 0x0F));
            s = _mm256_add_epi32(s,
                 _mm256_blend_epi32(_mm256_permutevar8x32_epi32(s, idx2), zeroi, 0x03));
            s = _mm256_add_epi32(s,
                 _mm256_blend_epi32(_mm256_permutevar8x32_epi32(s, idx1), zeroi, 0x01));
            s = _mm256_add_epi32(s, _mm256_set1_epi32(local_sum));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(prefix_sum + i), s);
            local_sum = _mm256_extract_epi32(s, 7);
        }
        // 尾部
        for (; i < end; ++i) {
            local_sum += weights[i];
            prefix_sum[i] = local_sum;
        }
        block_sums[b] = local_sum;
    }
}

// ---- 第二部分（b）：把全局块偏移加到 prefix_sum ----
inline void categorical_add_local_offsets(int* prefix_sum, size_t n, const int* block_sums,
                                          int tid, int nt) {
    constexpr size_t block_size = 2048;
    const size_t num_blocks = (n + block_size - 1) / block_size;
    size_t b0 = (num_blocks * (size_t)tid) / (size_t)nt;
    size_t b1 = (num_blocks * (size_t)(tid + 1)) / (size_t)nt;

    int off = 0;
    for (size_t t = 0; t < b0; ++t) off += block_sums[t];
    for (size_t b = b0; b < b1; ++b) {
        size_t start = b * block_size;
        size_t end = std::min(start + block_size, n);
        for (size_t i = start; i < end; ++i) prefix_sum[i] += off;
        off += block_sums[b];
    }
}

inline int categorical_total_weight(const int* prefix_sum, size_t n) {
    return prefix_sum[n - 1];
}

// 采样：r ∈ [0, total_weight)，upper_bound 在包含型前缀和上
// 返回第一个 > r 的位置，恰为「前缀 <= r < 当前前缀」的 token。
// 调用方需保证 total_weight > 0 且 n >= 1。
inline size_t categorical_sample(const int* prefix_sum, size_t n, int total_weight, uint64_t random_value) {
    int r = static_cast<int>(random_value % static_cast<uint64_t>(total_weight));
    return std::upper_bound(prefix_sum, prefix_sum + n, r) - prefix_sum;
}
