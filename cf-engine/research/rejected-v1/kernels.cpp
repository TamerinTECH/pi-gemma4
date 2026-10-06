#include "kernels.h"
#include "asm_kernel.h"

#include <cstring>

#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
#include <arm_neon.h>
#define CF_HAS_ARM_DOTPROD_CODE 1
#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(__clang__)
#define CF_DOTPROD_TARGET __attribute__((target("dotprod")))
#else
#define CF_DOTPROD_TARGET __attribute__((target("+dotprod")))
#endif
#else
#define CF_HAS_ARM_DOTPROD_CODE 0
#endif

namespace cf {
namespace {

inline uint16_t load_le16(const uint8_t * p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

// Exact IEEE binary16 -> binary32, including subnormals and signed zero.
// memcpy avoids aliasing violations and no host FP16 ISA is required.
inline float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 31u;
    uint32_t fraction = h & 1023u;
    uint32_t bits;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((fraction & 1024u) == 0) {
                fraction <<= 1;
                --e;
            }
            bits = sign | (uint32_t(e + 127) << 23) |
                   ((fraction & 1023u) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (fraction << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline void clear4(float out[4]) {
    out[0] = out[1] = out[2] = out[3] = 0.0f;
}

#if CF_HAS_ARM_DOTPROD_CODE
CF_DOTPROD_TARGET
inline int32x4_t dot_q4_row(const uint8_t * packed_row,
                           int8x16_t xlo, int8x16_t xhi,
                           uint8x16_t mask, int8x16_t offset) {
    const uint8x16_t packed = vld1q_u8(packed_row);
    const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, mask)), offset);
    const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), offset);
    int32x4_t sum = vdotq_s32(vdupq_n_s32(0), lo, xlo);
    return vdotq_s32(sum, hi, xhi);
}

CF_DOTPROD_TARGET
void dot4_arm_impl(const uint8_t * tile, const Q8Block * x,
                   size_t blocks, float out[4]) {
    // Candidate v2: one lane per output row, retained across all K blocks.
    // Conversion from binary16 to binary32 (FCVTL) is baseline AArch64 ASIMD;
    // no FP16 arithmetic, i8mm, SVE, or feature beyond dotprod is introduced.
    float32x4_t accum = vdupq_n_f32(0.0f);
    const uint8x16_t mask = vdupq_n_u8(15);
    const int8x16_t offset = vdupq_n_s8(8);
    for (size_t b = 0; b < blocks; ++b, tile += 72) {
        const int8x16_t xlo = vld1q_s8(x[b].qs);
        const int8x16_t xhi = vld1q_s8(x[b].qs + 16);
        // The activation vectors and their scale are shared by all four rows.
        // Expanded nibbles live in registers only; no expanded weight copy.
        const int32x4_t row0 = dot_q4_row(tile + 8, xlo, xhi, mask, offset);
        const int32x4_t row1 = dot_q4_row(tile + 24, xlo, xhi, mask, offset);
        const int32x4_t row2 = dot_q4_row(tile + 40, xlo, xhi, mask, offset);
        const int32x4_t row3 = dot_q4_row(tile + 56, xlo, xhi, mask, offset);
        const int32x4_t dots = vpaddq_s32(vpaddq_s32(row0,row1), vpaddq_s32(row2,row3));

        // Byte load preserves the public unaligned-tile contract without
        // dereferencing a possibly unaligned uint16_t pointer.
        uint8x8_t scale_bytes = vld1_u8(tile);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        scale_bytes = vrev16_u8(scale_bytes);
#endif
        const float32x4_t dw = vcvt_f32_f16(vreinterpret_f16_u16(vreinterpret_u16_u8(scale_bytes)));
        const float32x4_t dx = vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(x[b].d)));
        const float32x4_t scales = vmulq_f32(dw,dx);
        // K-block order is unchanged. Fused accumulation is checked against
        // the existing numerical tolerance; bit identity is not promised.
        accum = vfmaq_f32(accum,scales,vcvtq_f32_s32(dots));
    }
    vst1q_f32(out,accum);
}
#endif

} // namespace

bool has_dotprod() {
    static const bool supported = [] {
#if CF_HAS_ARM_DOTPROD_CODE && defined(__linux__) && defined(HWCAP_ASIMDDP)
        return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif CF_HAS_ARM_DOTPROD_CODE && defined(__APPLE__)
        int present = 0;
        size_t size = sizeof(present);
        return sysctlbyname("hw.optional.arm.FEAT_DotProd", &present, &size, nullptr, 0) == 0 && present != 0;
#else
        return false;
#endif
    }();
    return supported;
}

bool has_asm() {
#if defined(CF_ENABLE_ASM) && CF_ENABLE_ASM && defined(__aarch64__) && defined(__linux__) && !defined(__AARCH64EB__)
    return has_dotprod();
#else
    return false;
#endif
}

AsmKernelFn resolve_asm(AsmVariant variant) {
#if defined(CF_ENABLE_ASM) && CF_ENABLE_ASM && defined(__aarch64__) && defined(__linux__) && !defined(__AARCH64EB__)
    if (has_asm()) {
        switch (variant) {
            case AsmVariant::Baseline: return cf_q4_tile4_a76;
            case AsmVariant::Unroll2: return cf_q4_tile4_a76_unroll2;
            case AsmVariant::Prefetch512: return cf_q4_tile4_a76_prefetch512;
        }
    }
#else
    (void)variant;
#endif
    return nullptr;
}

void dot4_portable(const uint8_t * tile, const Q8Block * x,
                   size_t blocks, float out[4]) {
    clear4(out);
    for (size_t b = 0; b < blocks; ++b, tile += 72) {
        const float dx = half_to_float(x[b].d);
        for (size_t row = 0; row < 4; ++row) {
            const uint8_t * q = tile + 8 + row*16;
            int32_t dot = 0;
            for (size_t j = 0; j < 16; ++j) {
                dot += (int32_t(q[j] & 15)-8)*int32_t(x[b].qs[j]);
                dot += (int32_t(q[j] >> 4)-8)*int32_t(x[b].qs[j+16]);
            }
            const float scale = half_to_float(load_le16(tile + row*2))*dx;
            out[row] += scale*float(dot);
        }
    }
}

void dot4_arm(const uint8_t * tile, const Q8Block * x,
              size_t blocks, float out[4]) {
#if CF_HAS_ARM_DOTPROD_CODE
    if (has_dotprod()) {
        dot4_arm_impl(tile, x, blocks, out);
        return;
    }
#endif
    dot4_portable(tile, x, blocks, out);
}

void dot4_asm(const uint8_t * tile, const Q8Block * x,
              size_t blocks, float out[4], AsmVariant variant) {
    if (const auto kernel = resolve_asm(variant)) {
        kernel(tile,x,blocks,out);
        return;
    }
    dot4_portable(tile,x,blocks,out);
}

void build_lut(const Q8Block * x, size_t blocks, int16_t * lut) {
    for (size_t b = 0; b < blocks; ++b) {
        for (size_t group = 0; group < 8; ++group, lut += 16) {
            lut[0] = 0;
            // Each stage extends existing subset sums by one input element.
            // int16 is required: four int8 activations can sum to -512.
            for (size_t bit = 0; bit < 4; ++bit) {
                const size_t count = size_t(1) << bit;
                const int16_t add = x[b].qs[group*4 + bit];
                for (size_t mask = 0; mask < count; ++mask) {
                    lut[count + mask] = int16_t(lut[mask] + add);
                }
            }
        }
    }
}

void dot4_lut(const uint8_t * tile, const Q8Block * x, const int16_t * lut,
              size_t blocks, float out[4]) {
    clear4(out);
    for (size_t b = 0; b < blocks; ++b, tile += 72, lut += 128) {
        int32_t sums[4] = {0, 0, 0, 0};
        for (size_t group = 0; group < 8; ++group) {
            const int16_t * table = lut + group*16;
            const int32_t signed_offset = 8*int32_t(table[15]);
            for (size_t row = 0; row < 4; ++row) {
                const uint8_t * planes = tile + 8 + row*16 + group*2;
                sums[row] += int32_t(table[planes[0] & 15])
                           + 2*int32_t(table[planes[0] >> 4])
                           + 4*int32_t(table[planes[1] & 15])
                           + 8*int32_t(table[planes[1] >> 4]) - signed_offset;
            }
        }
        const float dx = half_to_float(x[b].d);
        for (size_t row = 0; row < 4; ++row) {
            const float scale = half_to_float(load_le16(tile + row*2))*dx;
            out[row] += scale*float(sums[row]);
        }
    }
}

} // namespace cf
