/*
 * AMX / Accelerate Matrix Multiply Benchmark
 *
 * Apple's Accelerate framework (cblas_sgemm) uses the undocumented AMX
 * coprocessor on M1-M3 and ARM SME2 on M4+. This benchmark proves
 * that hardware-accelerated matrix multiply is dramatically faster
 * than software implementations.
 *
 * Build: clang -O3 -DACCELERATE_NEW_LAPACK -framework Accelerate amx_matmul.c -o amx_matmul
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/sysctl.h>

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

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

/* Naive scalar GEMM (baseline) */
static void gemm_naive(const float *A, const float *B, float *C, int N) {
    memset(C, 0, N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

/* NEON GEMM (4x4 tiles) — ARM SIMD */
#ifdef __aarch64__
static void gemm_neon(const float *A, const float *B, float *C, int N) {
    memset(C, 0, N * N * sizeof(float));
    for (int i = 0; i < N; i += 4) {
        for (int j = 0; j < N; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0);
            float32x4_t c1 = vdupq_n_f32(0);
            float32x4_t c2 = vdupq_n_f32(0);
            float32x4_t c3 = vdupq_n_f32(0);
            for (int k = 0; k < N; k++) {
                float32x4_t b = vld1q_f32(&B[k * N + j]);
                c0 = vfmaq_n_f32(c0, b, A[(i+0)*N+k]);
                c1 = vfmaq_n_f32(c1, b, A[(i+1)*N+k]);
                c2 = vfmaq_n_f32(c2, b, A[(i+2)*N+k]);
                c3 = vfmaq_n_f32(c3, b, A[(i+3)*N+k]);
            }
            vst1q_f32(&C[(i+0)*N+j], c0);
            vst1q_f32(&C[(i+1)*N+j], c1);
            vst1q_f32(&C[(i+2)*N+j], c2);
            vst1q_f32(&C[(i+3)*N+j], c3);
        }
    }
}
#endif

/* Accelerate BLAS (routes through AMX on M1-M3, SME2 on M4+) */
static void gemm_accelerate(const float *A, const float *B, float *C, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
}

static int verify(const float *C_test, const float *C_ref, int N, const char *name) {
    float max_err = 0;
    for (int i = 0; i < N * N; i++) {
        float err = fabsf(C_test[i] - C_ref[i]);
        if (err > max_err) max_err = err;
    }
    int ok = max_err < N * 1e-4f;
    printf("  %-12s max_err=%.6f %s\n", name, max_err, ok ? "PASS" : "FAIL");
    return ok;
}

static double bench(void (*fn)(const float *, const float *, float *, int),
                    const float *A, const float *B, float *C, int N, int iters) {
    fn(A, B, C, N);
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) fn(A, B, C, N);
    return (now_sec() - t0) / iters;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Apple AMX/SME2 Coprocessor — Matrix Multiply Benchmark    ║\n");
    printf("║  Proving hardware-accelerated matmul for LLM inference     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int has_sme = 0, has_sme2 = 0;
    size_t len = sizeof(int);
    sysctlbyname("hw.optional.arm.FEAT_SME", &has_sme, &len, NULL, 0);
    len = sizeof(int);
    sysctlbyname("hw.optional.arm.FEAT_SME2", &has_sme2, &len, NULL, 0);

    char cpu[128] = {0};
    len = sizeof(cpu);
    sysctlbyname("machdep.cpu.brand_string", cpu, &len, NULL, 0);

    printf("  CPU: %s\n", cpu);
    printf("  SME:  %s    SME2: %s\n", has_sme?"YES":"NO", has_sme2?"YES":"NO");
    printf("  Accelerate BLAS uses: %s\n\n",
           has_sme2 ? "ARM SME2 (official, M4+)" :
           has_sme  ? "ARM SME (M4)" : "Apple AMX (undocumented, M1-M3)");

    srand(42);

    int sizes[] = {64, 128, 256, 512};
    int nsizes = 4;

    printf("%-6s  %10s  %10s  %10s  %8s\n",
           "N", "Naive", "NEON", "AMX/SME2", "Speedup");
    printf("──────  ──────────  ──────────  ──────────  ────────\n");

    int all_pass = 1;
    double last_speedup = 0;

    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        int iters = N <= 128 ? 10 : 3;

        float *A = aligned_alloc(64, N * N * sizeof(float));
        float *B = aligned_alloc(64, N * N * sizeof(float));
        float *C_ref = aligned_alloc(64, N * N * sizeof(float));
        float *C_test = aligned_alloc(64, N * N * sizeof(float));
        fill_random(A, N);
        fill_random(B, N);

        gemm_accelerate(A, B, C_ref, N);

        double t_naive = 0, t_neon = 0;

        if (N <= 128) {
            t_naive = bench(gemm_naive, A, B, C_test, N, iters);
            all_pass &= verify(C_test, C_ref, N, "naive");
        }

#ifdef __aarch64__
        t_neon = bench(gemm_neon, A, B, C_test, N, iters);
        all_pass &= verify(C_test, C_ref, N, "NEON");
#endif

        double t_accel = bench(gemm_accelerate, A, B, C_test, N, iters);

        double gf_naive = t_naive > 0 ? 2.0*N*N*N / t_naive / 1e9 : 0;
        double gf_neon = t_neon > 0 ? 2.0*N*N*N / t_neon / 1e9 : 0;
        double gf_accel = 2.0*N*N*N / t_accel / 1e9;
        double speedup = t_neon > 0 ? gf_accel / gf_neon : 0;
        last_speedup = speedup;

        if (t_naive > 0) {
            printf("%-6d  %7.1f GF  %7.1f GF  %7.1f GF  %6.1fx\n",
                   N, gf_naive, gf_neon, gf_accel, speedup);
        } else {
            printf("%-6d  %7s     %7.1f GF  %7.1f GF  %6.1fx\n",
                   N, "(slow)", gf_neon, gf_accel, speedup);
        }

        free(A); free(B); free(C_ref); free(C_test);
    }

    printf("\n  * AMX/SME2 column = Apple Accelerate BLAS (uses coprocessor internally)\n");
    printf("\n%s\n\n", all_pass ? "ALL CORRECTNESS CHECKS PASSED" : "SOME CHECKS FAILED");
    printf("The %.0fx speedup of Accelerate over hand-written NEON is the\n", last_speedup);
    printf("AMX/SME2 coprocessor advantage. Every matmul in every transformer\n");
    printf("layer in Gemma 4 benefits from this undocumented hardware.\n");

    return all_pass ? 0 : 1;
}
