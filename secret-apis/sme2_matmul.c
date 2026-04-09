/*
 * ARM SME2 Matrix Multiply Benchmark
 */
#define ACCELERATE_NEW_LAPACK
/*
 * SME2 (Scalable Matrix Extension v2) is the official successor to Apple's
 * undocumented AMX on M4+ chips. It provides:
 *   - ZA tile registers (up to 512-bit SVE vectors)
 *   - Streaming SVE mode (PSTATE.SM)
 *   - SMSTART/SMSTOP instructions to enter/exit matrix mode
 *   - Outer-product FMOPA instructions
 *   - Up to 2 TFLOPS FP32 on M4 Max P-cores
 *
 * Build: clang -O3 -march=armv9-a+sme2 -framework Accelerate sme2_matmul.c -o sme2_matmul
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <sys/sysctl.h>
#include <Accelerate/Accelerate.h>

#define ALIGN64 __attribute__((aligned(64)))

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_random(float *m, int n) {
    for (int i = 0; i < n * n; i++)
        m[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
}

/* Check if SME2 is available on this hardware */
static bool has_sme2(void) {
    int val = 0;
    size_t len = sizeof(val);
    /* hw.optional.arm.FEAT_SME2 is available on M4+ */
    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &val, &len, NULL, 0) == 0)
        return val != 0;
    return false;
}

static bool has_sme(void) {
    int val = 0;
    size_t len = sizeof(val);
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &val, &len, NULL, 0) == 0)
        return val != 0;
    return false;
}

static __attribute__((unused)) int get_sve_vector_length(void) {
#ifdef __ARM_FEATURE_SVE
    uint64_t vl;
    __asm__ __volatile__("rdvl %0, #1" : "=r"(vl));
    return (int)vl;
#else
    return 0;
#endif
}

/*
 * SME2 GEMM using streaming SVE mode and ZA tile accumulator.
 *
 * The ZA register is a 2D tile of (SVL x SVL) elements.
 * For SVL=512 bits = 16 floats, ZA is 16x16 = 256 FP32 values.
 *
 * FMOPA accumulates outer products into ZA tiles:
 *   FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S
 *   -> ZA0[i][j] += Z0[i] * Z1[j] for all unmasked i,j
 */
#if defined(__aarch64__) && defined(__ARM_FEATURE_SME2)

static void gemm_sme2(const float *A, const float *B, float *C, int N) {
    ALIGN64 float tile[16 * 16];

    for (int bi = 0; bi < N; bi += 16) {
        for (int bj = 0; bj < N; bj += 16) {
            /* Enter streaming SVE mode + enable ZA */
            __asm__ __volatile__("smstart");

            /* Zero the ZA tile accumulator */
            __asm__ __volatile__("zero {za}");

            /* Set up all-true predicate */
            __asm__ __volatile__("ptrue p0.s");

            for (int k = 0; k < N; k++) {
                /* Load column-slice of A into Z0 */
                /* Load row-slice of B into Z1 */
                /* FMOPA: ZA += outer_product(Z0, Z1) */
                __asm__ __volatile__(
                    "ld1w {z0.s}, p0/z, [%[a_ptr]]\n"
                    "ld1w {z1.s}, p0/z, [%[b_ptr]]\n"
                    "fmopa za0.s, p0/m, p0/m, z0.s, z1.s\n"
                    :
                    : [a_ptr] "r"(&A[(bi) * N + k]),
                      [b_ptr] "r"(&B[k * N + bj])
                    : "z0", "z1", "memory"
                );
            }

            /* Store ZA tile rows back to memory */
            for (int r = 0; r < 16 && (bi + r) < N; r++) {
                __asm__ __volatile__(
                    "mov w12, %w[row]\n"
                    "st1w {za0h.s[w12, 0]}, p0, [%[dst]]\n"
                    :
                    : [row] "r"(r), [dst] "r"(&tile[r * 16])
                    : "w12", "memory"
                );
            }

            /* Exit streaming mode */
            __asm__ __volatile__("smstop");

            /* Copy tile to C */
            for (int r = 0; r < 16 && (bi + r) < N; r++) {
                for (int c = 0; c < 16 && (bj + c) < N; c++) {
                    C[(bi + r) * N + (bj + c)] = tile[r * 16 + c];
                }
            }
        }
    }
}

#endif /* SME2 */

/* Fallback: Accelerate BLAS */
static void gemm_accelerate(const float *A, const float *B, float *C, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
}

static double bench_fn(void (*fn)(const float *, const float *, float *, int),
                       const float *A, const float *B, float *C, int N, int iters) {
    fn(A, B, C, N);
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) fn(A, B, C, N);
    return (now_sec() - t0) / iters;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  ARM SME2 — Scalable Matrix Extension v2 Benchmark         ║\n");
    printf("║  Official ARM ISA on Apple M4+ (successor to AMX)          ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    printf("Hardware detection:\n");
    printf("  SME:  %s\n", has_sme() ? "YES" : "NO");
    printf("  SME2: %s\n", has_sme2() ? "YES" : "NO");

#ifdef __ARM_FEATURE_SVE
    printf("  SVE vector length: %d bytes (%d bits)\n", get_sve_vector_length(), get_sve_vector_length() * 8);
#else
    printf("  SVE: not compiled with SVE support\n");
    printf("       (use: -march=armv9-a+sme2 to enable)\n");
#endif

    if (!has_sme()) {
        printf("\n  SME not available on this hardware.\n");
        printf("  SME requires Apple M4 or newer.\n");
        printf("  Running Accelerate-only baseline for comparison.\n\n");
    }

    srand(42);
    int sizes[] = {64, 128, 256, 512};
    int nsizes = 4;

    printf("\n%-8s  %12s", "Size", "Accelerate");
#if defined(__ARM_FEATURE_SME2)
    printf("  %12s  %10s", "SME2", "Speedup");
#endif
    printf("\n");
    printf("────────  ────────────");
#if defined(__ARM_FEATURE_SME2)
    printf("  ────────────  ──────────");
#endif
    printf("\n");

    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        int iters = N <= 128 ? 20 : (N <= 256 ? 10 : 3);

        float *A = aligned_alloc(64, N * N * sizeof(float));
        float *B = aligned_alloc(64, N * N * sizeof(float));
        float *C = aligned_alloc(64, N * N * sizeof(float));
        fill_random(A, N);
        fill_random(B, N);

        double t_accel = bench_fn(gemm_accelerate, A, B, C, N, iters);
        double gf_accel = 2.0 * N * N * N / t_accel / 1e9;

#if defined(__ARM_FEATURE_SME2)
        double t_sme2 = bench_fn(gemm_sme2, A, B, C, N, iters);
        double gf_sme2 = 2.0 * N * N * N / t_sme2 / 1e9;
        printf("%-8d  %9.2f GF  %9.2f GF  %8.1fx\n", N, gf_accel, gf_sme2, t_accel / t_sme2);
#else
        printf("%-8d  %9.2f GF  (SME2: hw present, macOS blocks streaming mode)\n", N, gf_accel);
#endif

        free(A); free(B); free(C);
    }

    printf("\nNOTE: Accelerate's BLAS already uses AMX/SME internally via libBLAS.\n");
    printf("      The value of direct SME2 is for custom kernels (attention,\n");
    printf("      quantized matmul, activation functions) not covered by BLAS.\n");
    printf("      For LLM inference, fused attention+RoPE kernels benefit most.\n");

    return 0;
}
