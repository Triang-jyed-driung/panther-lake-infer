#ifndef GEMM_FP16_H_
#define GEMM_FP16_H_

#include <cstddef>
#include <cstdint>

// ============================================================
// FP16 GEMM — Templated Interface
//
// C = A * B   (all arithmetic accumulated in FP32 internally)
//
// Matrices are row-major by default.
// A_transpose: A is stored as K×M instead of M×K
// B_transpose: B is stored as N×K instead of K×N
//
// Supported types:
//   TA, TB, TC ∈ { float, uint16_t, unsigned short }
//   float        → FP32  (standard IEEE 754 single precision)
//   uint16_t     → FP16  (stored in low 16 bits of each element)
// ============================================================

template <typename TA, typename TB, typename TC>
void gemm(size_t M, size_t N, size_t K,
          const TA *A, const TB *B, TC *C,
          bool A_transpose = false, bool B_transpose = false);

// ============================================================
// Legacy named functions (backward compatibility)
// ============================================================

// C[M][N] = A[M][K] * B[K][N]   —  all FP16
void fp16_gemm(size_t M, size_t N, size_t K,
               const unsigned short *A,
               const unsigned short *B,
               unsigned short *C,
               bool A_transpose = false,
               bool B_transpose = false);

// A: FP32,  B: FP16,  C: FP16
void fp32_fp16_gemm(size_t M, size_t N, size_t K,
                    const float *A,
                    const unsigned short *B,
                    unsigned short *C,
                    bool A_transpose = false,
                    bool B_transpose = false);

// A: FP16,  B: FP32,  C: FP16
void fp16_fp32_gemm(size_t M, size_t N, size_t K,
                    const unsigned short *A,
                    const float *B,
                    unsigned short *C,
                    bool A_transpose = false,
                    bool B_transpose = false);

// Thread control
void gemm_fp16_set_num_threads(int n);
int  gemm_fp16_get_num_threads();

#endif /* GEMM_FP16_H_ */
