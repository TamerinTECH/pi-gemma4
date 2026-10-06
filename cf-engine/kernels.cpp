#include "kernels.h"
#include "asm_kernel.h"

#include <cstring>
#include <cmath>

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

// Pinned A76 vec_dot's odd tail uses (float(dot)*weight_scale), followed by
// FMADD with activation_scale and the already-reduced even/odd accumulators.
inline void reference_tail(const uint8_t * tile, const Q8Block & x, float out[4]) {
    const float dx=half_to_float(x.d);
    for(size_t row=0;row<4;++row) {
        const uint8_t * q=tile+8+row*16;
        int32_t dot=0;
        for(size_t j=0;j<16;++j) {
            dot+=(int32_t(q[j]&15)-8)*int32_t(x.qs[j]);
            dot+=(int32_t(q[j]>>4)-8)*int32_t(x.qs[j+16]);
        }
        const float scaled=float(dot)*half_to_float(load_le16(tile+row*2));
        out[row]=std::fma(scaled,dx,out[row]);
    }
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
inline void reference_arm_block(const uint8_t * tile,const Q8Block & x,
                               float32x4_t & a0,float32x4_t & a1,
                               float32x4_t & a2,float32x4_t & a3) {
    const uint8x16_t mask = vdupq_n_u8(15);
    const int8x16_t offset = vdupq_n_s8(8);
    const int8x16_t xlo=vld1q_s8(x.qs),xhi=vld1q_s8(x.qs+16);
    const int32x4_t row0=dot_q4_row(tile+8,xlo,xhi,mask,offset);
    const int32x4_t row1=dot_q4_row(tile+24,xlo,xhi,mask,offset);
    const int32x4_t row2=dot_q4_row(tile+40,xlo,xhi,mask,offset);
    const int32x4_t row3=dot_q4_row(tile+56,xlo,xhi,mask,offset);
    uint8x8_t scale_bytes=vld1_u8(tile);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    scale_bytes=vrev16_u8(scale_bytes);
#endif
    const float32x4_t dw=vcvt_f32_f16(vreinterpret_f16_u16(vreinterpret_u16_u8(scale_bytes)));
    const float32x4_t dx=vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(x.d)));
    const float32x4_t scales=vmulq_f32(dw,dx);
    // Match actual pinned instructions: separate FMUL then FADD, preserving
    // each SDOT lane until the final horizontal reduction. Compile this file
    // with -ffp-contract=off; do not substitute FMLA or fast-math reassociation.
    a0=vaddq_f32(a0,vmulq_laneq_f32(vcvtq_f32_s32(row0),scales,0));
    a1=vaddq_f32(a1,vmulq_laneq_f32(vcvtq_f32_s32(row1),scales,1));
    a2=vaddq_f32(a2,vmulq_laneq_f32(vcvtq_f32_s32(row2),scales,2));
    a3=vaddq_f32(a3,vmulq_laneq_f32(vcvtq_f32_s32(row3),scales,3));
}

CF_DOTPROD_TARGET
void dot4_arm_impl(const uint8_t * tile,const Q8Block * x,size_t blocks,float out[4]) {
    float32x4_t e0=vdupq_n_f32(0),e1=e0,e2=e0,e3=e0;
    float32x4_t o0=e0,o1=e0,o2=e0,o3=e0;
    size_t b=0;
    for(;b+1<blocks;b+=2,tile+=144) {
        reference_arm_block(tile,x[b],e0,e1,e2,e3);
        reference_arm_block(tile+72,x[b+1],o0,o1,o2,o3);
    }
    const float32x4_t even=vpaddq_f32(vpaddq_f32(e0,e1),vpaddq_f32(e2,e3));
    const float32x4_t odd=vpaddq_f32(vpaddq_f32(o0,o1),vpaddq_f32(o2,o3));
    vst1q_f32(out,vaddq_f32(odd,even));
    if(b<blocks)reference_tail(tile,x[b],out);
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
    float accum[2][4][4]={};
    const size_t paired=blocks&~size_t(1);
    size_t b=0;
    for (;b<paired;++b,tile+=72) {
        const float dx = half_to_float(x[b].d);
        for (size_t row = 0; row < 4; ++row) {
            const uint8_t * q = tile + 8 + row*16;
            const float scale = half_to_float(load_le16(tile + row*2))*dx;
            for(size_t lane=0;lane<4;++lane) {
                int32_t dot=0;
                for(size_t offset=0;offset<4;++offset) {
                    const size_t j=lane*4+offset;
                    dot+=(int32_t(q[j]&15)-8)*int32_t(x[b].qs[j]);
                    dot+=(int32_t(q[j]>>4)-8)*int32_t(x[b].qs[j+16]);
                }
                const float product=float(dot)*scale;
                accum[b%2][row][lane]=accum[b%2][row][lane]+product;
            }
        }
    }
    for(size_t row=0;row<4;++row) {
        const auto * even=accum[0][row];const auto * odd=accum[1][row];
        const float es=(even[0]+even[1])+(even[2]+even[3]);
        const float os=(odd[0]+odd[1])+(odd[2]+odd[3]);
        out[row]=os+es;
    }
    if(b<blocks)reference_tail(tile,x[b],out);
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
    float accum[2][4][4]={};
    const size_t paired=blocks&~size_t(1);
    for (size_t b=0;b<blocks;++b,tile+=72,lut+=128) {
        int32_t sums[4][4]={};
        for (size_t group = 0; group < 8; ++group) {
            const int16_t * table = lut + group*16;
            const int32_t signed_offset = 8*int32_t(table[15]);
            for (size_t row = 0; row < 4; ++row) {
                const uint8_t * planes = tile + 8 + row*16 + group*2;
                // Groups 0..3 and 4..7 are the low/high halves of the same
                // four SDOT lanes; keep them separate until float reduction.
                sums[row][group%4] += int32_t(table[planes[0] & 15])
                           + 2*int32_t(table[planes[0] >> 4])
                           + 4*int32_t(table[planes[1] & 15])
                           + 8*int32_t(table[planes[1] >> 4]) - signed_offset;
            }
        }
        const float dx = half_to_float(x[b].d);
        for (size_t row = 0; row < 4; ++row) {
            const float dw=half_to_float(load_le16(tile+row*2));
            if(b<paired) {
                const float scale=dw*dx;
                for(size_t lane=0;lane<4;++lane) {
                    const float product=float(sums[row][lane])*scale;
                    accum[b%2][row][lane]=accum[b%2][row][lane]+product;
                }
            } else {
                const auto * even=accum[0][row];const auto * odd=accum[1][row];
                const float es=(even[0]+even[1])+(even[2]+even[3]);
                const float os=(odd[0]+odd[1])+(odd[2]+odd[3]);
                const int32_t dot=sums[row][0]+sums[row][1]+sums[row][2]+sums[row][3];
                out[row]=std::fma(float(dot)*dw,dx,os+es);
            }
        }
    }
    if(blocks==paired)for(size_t row=0;row<4;++row) {
        const auto * even=accum[0][row];const auto * odd=accum[1][row];
        const float es=(even[0]+even[1])+(even[2]+even[3]);
        const float os=(odd[0]+odd[1])+(odd[2]+odd[3]);
        out[row]=os+es;
    }
}

} // namespace cf
