#pragma once

#include <cstddef>
#include <cstdint>

namespace cf {

enum class Layout { Tile4, Bitplane4 };
enum class AsmVariant { Baseline, Unroll2, Prefetch512 };

// Exactly the GGML Q8_0 wire representation. d is an IEEE binary16 bit pattern.
struct Q8Block {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(Q8Block) == 34, "Q8_0 blocks must occupy exactly 34 bytes");
static_assert(offsetof(Q8Block, qs) == 2, "Q8_0 scale must precede its 32 codes");

// True only when this binary contains the ARM implementation AND the running
// CPU advertises dot product. Callers must report false as a portable fallback.
bool has_dotprod();
// Assembly exists only when built with CF_ENABLE_ASM and its .S object linked.
// Also checks runtime dotprod support. No unchecked ISA call is exposed here.
bool has_asm();
using AsmKernelFn = void (*)(const uint8_t *, const void *, size_t, float *);
// Resolve once outside the tile loop. nullptr means an explicit fallback is
// required; a nonnull result has already passed the runtime feature check.
AsmKernelFn resolve_asm(AsmVariant variant = AsmVariant::Baseline);

// No allocation or synchronization in any kernel. Each tile is blocks*72 bytes.
// Unaligned tile addresses are supported. x must be suitably aligned for Q8Block.
// Outputs are overwritten, including for blocks == 0. Pointers must otherwise
// address the complete input. Callers are responsible for finite numerical
// inputs; the current compiler/adapter checks integrity/bounds, not finiteness.
// Tile4 implementations preserve the pinned A76 reference's four SDOT lanes,
// separate even/odd block chains and unfused multiply/add order. kernels.cpp
// must compile with -ffp-contract=off and without reassociative fast math.
void build_lut(const Q8Block * x, size_t blocks, int16_t * lut);
void dot4_portable(const uint8_t * tile, const Q8Block * x,
                   size_t blocks, float out[4]);
// Safely dispatches to the portable implementation if has_dotprod() is false.
void dot4_arm(const uint8_t * tile, const Q8Block * x,
              size_t blocks, float out[4]);
void dot4_asm(const uint8_t * tile, const Q8Block * x,
              size_t blocks, float out[4], AsmVariant variant = AsmVariant::Baseline);
void dot4_lut(const uint8_t * tile, const Q8Block * x, const int16_t * lut,
              size_t blocks, float out[4]);

} // namespace cf
