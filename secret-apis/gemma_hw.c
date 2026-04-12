/*
 * libgemma_hw — Accelerate SGEMM entry point (public C ABI).
 */

#include "gemma_hw.h"
#include <string.h>

#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#endif

int gemma_hw_sgemm_row_major(int m, int n, int k,
                              float alpha, const float *a, int lda,
                              const float *b, int ldb,
                              float beta, float *c, int ldc)
{
#if defined(__APPLE__)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    return 0;
#else
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)beta;
    (void)c;
    (void)ldc;
    return -1;
#endif
}
