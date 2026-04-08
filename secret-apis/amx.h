/*
 * Apple AMX (Apple Matrix eXtensions) — Undocumented Coprocessor Interface
 *
 * Reverse-engineered from dougallj's work and the corsix/amx project.
 * AMX is a matrix coprocessor on Apple Silicon (M1-M3) accessed via
 * undocumented ARM64 instructions encoded in the HINt space.
 *
 * On M4+, AMX is superseded by ARM SME2 but these instructions still work.
 * The coprocessor operates on dedicated register files:
 *   - X registers: 8 x 512-bit (64 bytes each)
 *   - Y registers: 8 x 512-bit
 *   - Z registers: 8 x 512-bit (accumulator)
 *
 * All instructions are encoded as: .word (0x00201000 | (op << 5) | operand_gpr)
 */

#ifndef AMX_H
#define AMX_H

#include <stdint.h>

#ifdef __aarch64__

/* AMX instruction encoding: HINt-space coprocessor command
 * Format: .word 0x00201000 | (amx_op << 5) | Xn
 * where Xn is the GPR holding the operand descriptor */
#define AMX_OP(op, gpr) \
    __asm__ __volatile__( \
        ".word (0x00201000 | (" #op " << 5) | 0%0)" \
        : : "i"(gpr) : "memory")

/* Convenience: pass operand in x0 */
static inline void amx_op_x0(uint32_t op) {
    __asm__ __volatile__(
        ".word (0x00201000 | (%w0 << 5) | 0)"
        : : "r"(op) : "memory"
    );
}

/* AMX opcodes — each operates on the coprocessor register file */
#define AMX_LDX    0   /* Load into X register */
#define AMX_LDY    1   /* Load into Y register */
#define AMX_STX    2   /* Store from X register */
#define AMX_STY    3   /* Store from Y register */
#define AMX_LDZ    4   /* Load into Z accumulator register */
#define AMX_STZ    5   /* Store from Z accumulator register */
#define AMX_LDZI   6   /* Load interleaved into Z */
#define AMX_STZI   7   /* Store interleaved from Z */
#define AMX_EXTRX  8   /* Extract from X register */
#define AMX_EXTRY  9   /* Extract from Y register */
#define AMX_FMA64  10  /* Fused multiply-accumulate f64 */
#define AMX_FMS64  11  /* Fused multiply-subtract f64 */
#define AMX_FMA32  12  /* Fused multiply-accumulate f32 */
#define AMX_FMS32  13  /* Fused multiply-subtract f32 */
#define AMX_MAC16  14  /* Integer multiply-accumulate 16-bit */
#define AMX_FMA16  15  /* Fused multiply-accumulate f16 */
#define AMX_FMS16  16  /* Fused multiply-subtract f16 */
#define AMX_SET    17  /* AMX configuration (enable/disable) */
#define AMX_VECINT 18  /* Vector integer op */
#define AMX_VECFP  19  /* Vector floating-point op */
#define AMX_MATINT 20  /* Matrix integer op */
#define AMX_MATFP  21  /* Matrix floating-point op */
#define AMX_GENLUT 22  /* Generate lookup table */

/* Enable/disable the AMX coprocessor */
static inline void amx_enable(void) {
    /* SET mode 0 = enable */
    uint64_t v = 0;
    __asm__ __volatile__(
        ".word (0x00201000 | (17 << 5) | 0)\n"
        : : : "memory"
    );
    (void)v;
}

static inline void amx_disable(void) {
    /* SET mode 1 = disable */
    register uint64_t v __asm__("x0") = 1;
    __asm__ __volatile__(
        ".word (0x00201000 | (17 << 5) | 0)\n"
        : : "r"(v) : "memory"
    );
}

/*
 * AMX operand descriptor format for load/store:
 *   bits [63:56] = register pair index (which X/Y/Z register)
 *   bits [55:0]  = memory address (56-bit physical pointer)
 *
 * For FMA operations:
 *   bits [63:58] = Z row offset
 *   bits [57:48] = Y offset
 *   bits [47:38] = X offset
 *   bits [37:32] = operation flags
 *   bits [31:20] = count/mask
 */

/* Build a load/store descriptor: register index + memory address */
static inline uint64_t amx_ldst_operand(uint8_t reg_idx, const void *addr) {
    return ((uint64_t)reg_idx << 56) | ((uint64_t)(uintptr_t)addr & 0x00FFFFFFFFFFFFFFULL);
}

/* Build an FMA descriptor for outer-product accumulation */
static inline uint64_t amx_fma_operand(uint8_t z_row, uint8_t y_off, uint8_t x_off) {
    return ((uint64_t)z_row << 20) | ((uint64_t)x_off << 10) | (uint64_t)y_off;
}

/* Load 64 bytes from memory into X register `reg` */
static inline void amx_ldx(uint8_t reg, const void *addr) {
    register uint64_t operand __asm__("x0") = amx_ldst_operand(reg, addr);
    __asm__ __volatile__(
        ".word (0x00201000 | (0 << 5) | 0)\n"
        : : "r"(operand) : "memory"
    );
}

/* Load 64 bytes from memory into Y register `reg` */
static inline void amx_ldy(uint8_t reg, const void *addr) {
    register uint64_t operand __asm__("x0") = amx_ldst_operand(reg, addr);
    __asm__ __volatile__(
        ".word (0x00201000 | (1 << 5) | 0)\n"
        : : "r"(operand) : "memory"
    );
}

/* Store 64 bytes from Z register `reg` to memory */
static inline void amx_stz(uint8_t reg, void *addr) {
    register uint64_t operand __asm__("x0") = amx_ldst_operand(reg, addr);
    __asm__ __volatile__(
        ".word (0x00201000 | (5 << 5) | 0)\n"
        : : "r"(operand) : "memory"
    );
}

/* Load 64 bytes into Z register `reg` from memory (to init accumulators) */
static inline void amx_ldz(uint8_t reg, const void *addr) {
    register uint64_t operand __asm__("x0") = amx_ldst_operand(reg, addr);
    __asm__ __volatile__(
        ".word (0x00201000 | (4 << 5) | 0)\n"
        : : "r"(operand) : "memory"
    );
}

/* FP32 fused multiply-accumulate: Z += outer_product(X[x_off], Y[y_off]) */
static inline void amx_fma32(uint64_t operand) {
    register uint64_t op __asm__("x0") = operand;
    __asm__ __volatile__(
        ".word (0x00201000 | (12 << 5) | 0)\n"
        : : "r"(op) : "memory"
    );
}

/* FP16 fused multiply-accumulate */
static inline void amx_fma16(uint64_t operand) {
    register uint64_t op __asm__("x0") = operand;
    __asm__ __volatile__(
        ".word (0x00201000 | (15 << 5) | 0)\n"
        : : "r"(op) : "memory"
    );
}

#else
/* Stub for non-aarch64 — these are Apple Silicon only */
static inline void amx_enable(void) {}
static inline void amx_disable(void) {}
#endif /* __aarch64__ */

#endif /* AMX_H */
