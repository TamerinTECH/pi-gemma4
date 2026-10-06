#pragma once

#include "kernels.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cf::kleidi {

// Direct, disclosed external-kernel comparison; never a silent CF fallback.
// Original rows: [m,k/32] GGML Q4_0 blocks (fp16 d, 16 unsigned nibble bytes).
struct Weights {
    size_t k = 0, m = 0, source_bytes = 0;
    std::vector<uint8_t> packed;
    double pack_ms = 0, validation_ms = 0;
    bool roundtrip = false;
};

struct Activation {
    size_t k = 0;
    std::vector<uint8_t> packed;
    double pack_ms = 0;
    const char * quantization = "uninitialized";
};

// Requires compiled AArch64 dotprod kernel and runtime ASIMD/dotprod/FP16.
bool available();
const char * kernel_name();
Weights prepare(const uint8_t * original_q4_rows, size_t k, size_t m);
Activation prepare_activation(size_t k);

// Uses KleidiAI's own generic qsi8d32p quantizer, roundf (ties away).
// This may differ from GGML's AArch64 nearest-even Q8_0 quantizer.
void pack_activation(Activation & activation, const float * x);
// Same activation codes/scales as the reference; no requantization.
// For mr=1 the Kleidi input is byte-identical to GGML Q8_0.
void pack_q8(Activation & activation, const Q8Block * x);

// Read-only and allocation-free. Caller owns scheduling and output buffer.
// row_start and row_count are multiples of 4; count=0 means remaining rows.
// out points at the FULL m-element output, and this call writes the named range.
// Throws if unavailable, malformed, or activation was not packed; no fallback.
void run(const Weights & weights, const Activation & activation, float * out,
         size_t row_start = 0, size_t row_count = 0);

} // namespace cf::kleidi
