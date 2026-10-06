#include "kleidi_compare.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#if defined(CF_KLEIDI_ENABLED)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#endif

namespace cf::kleidi {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
void require(bool ok, const char * message) {
    if (!ok) throw std::invalid_argument(message);
}
size_t multiply(size_t a, size_t b) {
    require(!a || b <= SIZE_MAX/a, "Kleidi dimension overflow");
    return a*b;
}
void check_k(size_t k) {
    require(k && k%32 == 0, "Kleidi K must be a positive multiple of 32");
    const uint16_t one = 1;
    require(*reinterpret_cast<const uint8_t *>(&one) == 1, "Kleidi comparison requires little endian");
}
void check_activation(const Activation & a) {
    check_k(a.k);
    require(a.packed.size() == multiply(a.k/32, 34), "Kleidi activation size mismatch");
}
bool finite_half(const uint8_t * data) {
    uint16_t bits; std::memcpy(&bits, data, sizeof(bits));
    return (bits & 0x7c00) != 0x7c00;
}
} // namespace

bool available() {
#if defined(CF_KLEIDI_ENABLED)
    static const bool ok = [] {
        const unsigned long flags = getauxval(AT_HWCAP);
        const unsigned long required = HWCAP_ASIMD | HWCAP_ASIMDDP | HWCAP_FPHP | HWCAP_ASIMDHP;
        return (flags & required) == required;
    }();
    return ok;
#else
    return false;
#endif
}

const char * kernel_name() {
    return "kleidiai-v1.24.0-qsi8d32p1x8-qsi4c32p4x8-1x4x32-neon-dotprod";
}

Weights prepare(const uint8_t * original, size_t k, size_t m) {
    check_k(k);
    require(original && m && m%4 == 0, "Kleidi M must be a positive multiple of 4");
    Weights result; result.k = k; result.m = m;
    const size_t blocks = k/32, row_bytes = multiply(blocks, 18);
    result.source_bytes = multiply(m, row_bytes);
    const size_t packed_bytes = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(m,k,4,16,32);
    require(packed_bytes == result.source_bytes, "Kleidi packed bytes unexpectedly differ from Q4_0");
    result.packed.resize(packed_bytes);
    for (size_t i=0; i<m*blocks; ++i)
        require(finite_half(original+i*18), "Kleidi source Q4_0 scale is nonfinite");
    const auto start = Clock::now();
    const kai_rhs_pack_qs4cxs1s0_param params{1,8};
    kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(1,m,k,4,16,2,32,original,nullptr,result.packed.data(),0,&params);
    result.pack_ms = elapsed(start);
    const auto validation_start = Clock::now();
    // Inverse of pinned kr=16,sr=2 layout. Compare every source byte, including
    // negative/zero scale bit patterns and all integer codes. No expanded copy.
    for (size_t row=0; row<m; ++row) for (size_t b=0; b<blocks; ++b) {
        const auto src = original+row*row_bytes+b*18;
        const auto tile = result.packed.data()+((row/4)*blocks+b)*72;
        require(std::memcmp(src,tile+(row%4)*2,2)==0, "Kleidi scale roundtrip failed");
        for (size_t q=0; q<16; ++q) {
            const uint8_t decoded = tile[8+(q/8)*32+(row%4)*8+(q%8)]^0x88;
            require(decoded==src[2+q], "Kleidi integer-code roundtrip failed");
        }
    }
    result.roundtrip = true;
    result.validation_ms = elapsed(validation_start);
    return result;
}

Activation prepare_activation(size_t k) {
    check_k(k);
    Activation a; a.k=k; a.packed.resize(multiply(k/32,34));
    return a;
}

void pack_q8(Activation & a, const Q8Block * x) {
    check_activation(a); require(x, "Kleidi null Q8 activation");
    const auto start=Clock::now();
    for (size_t b=0; b<a.k/32; ++b)
        require((x[b].d&0x7c00)!=0x7c00, "Kleidi Q8 scale is nonfinite");
    std::memcpy(a.packed.data(),x,a.packed.size());
    a.quantization="same-ggml-q8_0-codes-and-scale-bits"; a.pack_ms=elapsed(start);
}

void pack_activation(Activation & a, const float * x) {
    check_activation(a); require(x, "Kleidi null float activation");
    require(available(), "Kleidi native activation packing unavailable on this CPU/build");
#if defined(CF_KLEIDI_ENABLED)
    const auto start=Clock::now();
    // Refuse a scale that cannot be represented by the required fp16 format.
    for (size_t i=0;i<a.k;++i)
        require(std::isfinite(x[i]) && std::abs(x[i])/127.0f<=65504.0f, "Kleidi activation cannot produce finite fp16 scale");
    kai_run_lhs_quant_pack_qsi8d32p_f32(1,a.k,32,1,16,2,0,x,a.k*sizeof(float),a.packed.data());
    a.quantization="kleidiai-native-roundf-ties-away"; a.pack_ms=elapsed(start);
#endif
}

void run(const Weights & w, const Activation & a, float * out, size_t first, size_t count) {
    require(available(), "Kleidi dotprod kernel unavailable on this CPU/build; no fallback");
    require(out && w.roundtrip && w.k==a.k, "Kleidi unprepared or mismatched operands");
    require(std::strcmp(a.quantization,"uninitialized")!=0, "Kleidi activation not packed");
    require(first<=w.m && first%4==0, "Kleidi row start invalid");
    if (!count) count=w.m-first;
    require(count<=w.m-first && count%4==0, "Kleidi row count invalid");
    if (!count) return;
#if defined(CF_KLEIDI_ENABLED)
    const size_t offset=kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(first,w.k,32);
    kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
        1,count,w.k,32,a.packed.data(),w.packed.data()+offset,out+first,w.m*sizeof(float),sizeof(float),
        -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity());
#endif
}
} // namespace cf::kleidi
