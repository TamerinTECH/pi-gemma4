#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Little-endian Linux AArch64 AAPCS64. Requires ASIMD and ASIMDDP; the C++ wrapper performs
// runtime feature dispatch. Q8 input is blocks consecutive 34-byte Q8_0 blocks.
// Weights are blocks consecutive 72-byte q4-tile4-v1 blocks. Output is four
// writable float32 values. Inputs may be unaligned; output must be float-aligned.
// No allocation, stack, syscalls, or accesses outside these bounded ranges.
// blocks==0 does not read either input and overwrites output with zero.
typedef void (*cf_asm_kernel_fn)(const uint8_t *, const void *, size_t, float *);
void cf_q4_tile4_a76(const uint8_t * tile, const void * q8, size_t blocks, float * out);
void cf_q4_tile4_a76_unroll2(const uint8_t * tile, const void * q8, size_t blocks, float * out);
void cf_q4_tile4_a76_prefetch512(const uint8_t * tile, const void * q8, size_t blocks, float * out);

// Test-only harness: returns a failure bitmask for callee-preserved x19-x29 and
// low 64 bits of v8-v15. The three arithmetic entrypoints themselves use no
// callee-preserved registers and never modify SP. This probe has a stack frame.
uint64_t cf_q4_tile4_a76_abi_probe(const uint8_t * tile, const void * q8,
                                 size_t blocks, float * out, cf_asm_kernel_fn kernel);

#ifdef __cplusplus
}
#endif
