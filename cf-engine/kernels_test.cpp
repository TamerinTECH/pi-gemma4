// Standalone: c++ -O3 -std=c++17 -pthread kernels.cpp kernels_test.cpp -o kernels_test
// Correctness: ./kernels_test
// Synthetic captured-shape timing: ./kernels_test --bench --iterations 5
// This is an operator test, never evidence of full-model throughput or DRAM bytes.
#include "kernels.h"
#include "asm_kernel.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
constexpr double kRelativeL2 = 2e-5;
constexpr double kMaximumAbsolute = 2e-4;
uint32_t random_code(uint32_t & state) {
    state = state*1664525u + 1013904223u;
    return state;
}
void require(bool condition, const char * what) {
    if (!condition) throw std::runtime_error(what);
}
uint16_t get16(const uint8_t * p) {
    return uint16_t(p[0]) | uint16_t(uint16_t(p[1]) << 8);
}
void put16(uint8_t * p, uint16_t value) {
    p[0] = uint8_t(value);
    p[1] = uint8_t(value >> 8);
}

// Independent mathematical binary16 conversion for the test oracle.
double reference_half(uint16_t bits) {
    const int exponent = (bits >> 10) & 31;
    const int mantissa = bits & 1023;
    const double magnitude = exponent == 0
        ? std::ldexp(double(mantissa), -24)
        : std::ldexp(double(mantissa + 1024), exponent - 25);
    require(exponent != 31, "test oracle accepts finite scales only");
    return (bits & 0x8000) ? -magnitude : magnitude;
}

// Original Q4_0 memory order: [row][block]{fp16 d, qs[16]}.
void pack_original(const std::vector<uint8_t> & source, size_t blocks,
                   uint8_t * tile, uint8_t * planes) {
    for (size_t b = 0; b < blocks; ++b) {
        std::memset(planes + b*72, 0, 72);
        for (size_t row = 0; row < 4; ++row) {
            const uint8_t * raw = source.data() + (row*blocks + b)*18;
            std::memcpy(tile + b*72 + row*2, raw, 2);
            std::memcpy(planes + b*72 + row*2, raw, 2);
            std::memcpy(tile + b*72 + 8 + row*16, raw + 2, 16);
            for (size_t j = 0; j < 32; ++j) {
                const uint8_t code = (raw[2 + j%16] >> (j < 16 ? 0 : 4)) & 15;
                for (size_t bit = 0; bit < 4; ++bit) {
                    const size_t index = b*72 + 8 + row*16 + (j/4)*2 + bit/2;
                    planes[index] |= uint8_t(((code >> bit) & 1) << (j%4 + (bit%2)*4));
                }
            }
        }
    }
}

void assert_roundtrip(const std::vector<uint8_t> & source, size_t blocks,
                      const uint8_t * tile, const uint8_t * planes) {
    for (size_t b = 0; b < blocks; ++b) {
        for (size_t row = 0; row < 4; ++row) {
            const uint8_t * raw = source.data() + (row*blocks + b)*18;
            require(std::memcmp(raw, tile + b*72 + row*2, 2) == 0, "tile scale roundtrip");
            require(std::memcmp(raw, planes + b*72 + row*2, 2) == 0, "plane scale roundtrip");
            require(std::memcmp(raw + 2, tile + b*72 + 8 + row*16, 16) == 0, "tile code roundtrip");
            for (size_t j = 0; j < 32; ++j) {
                uint8_t recovered = 0;
                for (size_t bit = 0; bit < 4; ++bit) {
                    const uint8_t packed = planes[b*72 + 8 + row*16 + (j/4)*2 + bit/2];
                    recovered |= uint8_t(((packed >> (j%4 + (bit%2)*4)) & 1) << bit);
                }
                require(recovered == ((raw[2 + j%16] >> (j < 16 ? 0 : 4)) & 15), "plane code roundtrip");
            }
        }
    }
}

std::array<double,4> reference(const std::vector<uint8_t> & raw,
                               const std::vector<cf::Q8Block> & x) {
    std::array<double,4> out{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t b = 0; b < x.size(); ++b) {
            const uint8_t * source = raw.data() + (row*x.size() + b)*18;
            const double weight_scale = reference_half(get16(source));
            const double activation_scale = reference_half(x[b].d);
            for (size_t j = 0; j < 32; ++j) {
                const int code = (source[2 + j%16] >> (j < 16 ? 0 : 4)) & 15;
                const double w = weight_scale*double(code - 8);
                const double a = activation_scale*double(x[b].qs[j]);
                out[row] += w*a;
            }
        }
    }
    return out;
}

struct Errors { double l2 = 0; double max_normalized = 0; };
Errors check(const std::array<double,4> & reference_value, const float out[4]) {
    double sum_error = 0, sum_reference = 0, max_error = 0, max_reference = 0;
    for (size_t row = 0; row < 4; ++row) {
        require(std::isfinite(out[row]), "nonfinite output");
        const double error = std::abs(double(out[row])-reference_value[row]);
        sum_error += error*error;
        sum_reference += reference_value[row]*reference_value[row];
        max_error = std::max(max_error, error);
        max_reference = std::max(max_reference, std::abs(reference_value[row]));
    }
    Errors errors;
    errors.l2 = sum_reference ? std::sqrt(sum_error/sum_reference) : std::sqrt(sum_error);
    errors.max_normalized = max_error/std::max(1.0, max_reference);
    require(errors.l2 <= kRelativeL2, "predeclared relative-L2 tolerance exceeded");
    require(errors.max_normalized <= kMaximumAbsolute, "predeclared maximum-absolute tolerance exceeded");
    return errors;
}

void correctness() {
    // Positive/negative zero, subnormal extrema, normal boundary, +/-1,
    // diverse fractional scales, and the largest finite binary16 magnitudes.
    const uint16_t scale_bits[] = {0,0x8000,1,0x8001,0x03ff,0x0400,0x3555,
                                  0x3c00,0xbc00,0x4201,0xc555,0x7bff,0xfbff};
    uint32_t state = 0xcfe12345u;
    size_t cases = 0;
    Errors worst;
    for (const size_t blocks : {size_t(0),size_t(1),size_t(2),size_t(3),size_t(22),size_t(44),size_t(88)}) {
        for (size_t mode = 0; mode < 7; ++mode) {
            std::vector<uint8_t> original(blocks*4*18);
            std::vector<cf::Q8Block> x(blocks);
            for (size_t b = 0; b < blocks; ++b) {
                x[b].d = scale_bits[(b + mode*3) % 13];
                for (size_t j = 0; j < 32; ++j) {
                    int value = int(random_code(state) >> 24)-128;
                    if (mode == 0) value = 0;
                    if (mode == 1) value = -128;
                    if (mode == 2) value = 127;
                    if (mode == 3) value = j%2 ? -128 : 127;
                    x[b].qs[j] = int8_t(value);
                }
                for (size_t row = 0; row < 4; ++row) {
                    uint8_t * raw = original.data() + (row*blocks + b)*18;
                    put16(raw, scale_bits[(b*3 + row + mode) % 13]);
                    for (size_t j = 0; j < 16; ++j) {
                        uint8_t value = uint8_t(random_code(state) >> 24);
                        if (mode == 1) value = 0;
                        if (mode == 2) value = 255;
                        if (mode == 3) value = uint8_t(j | ((15-j) << 4));
                        if (mode == 4) value = 0x88;
                        raw[2+j] = value;
                    }
                }
            }
            const auto expected = reference(original, x);
            std::vector<int16_t> lut(blocks*128);
            cf::build_lut(x.data(), blocks, lut.data());
            for (size_t b = 0; b < blocks; ++b) {
                for (size_t g = 0; g < 8; ++g) {
                    for (size_t mask = 0; mask < 16; ++mask) {
                        int sum = 0;
                        for (size_t bit = 0; bit < 4; ++bit)
                            if (mask & (size_t(1) << bit)) sum += x[b].qs[g*4 + bit];
                        require(lut[b*128 + g*16 + mask] == sum, "LUT subset sum");
                    }
                }
            }
            for (size_t alignment = 0; alignment < 16; ++alignment) {
                std::vector<uint8_t> tile(blocks*72 + 16), planes(blocks*72 + 16);
                pack_original(original, blocks, tile.data()+alignment, planes.data()+alignment);
                assert_roundtrip(original, blocks, tile.data()+alignment, planes.data()+alignment);
                float out[4];
                for (size_t method = 0; method < 6; ++method) {
                    if (method == 0) cf::dot4_portable(tile.data()+alignment,x.data(),blocks,out);
                    if (method == 1) cf::dot4_arm(tile.data()+alignment,x.data(),blocks,out);
                    if (method == 2) cf::dot4_lut(planes.data()+alignment,x.data(),lut.data(),blocks,out);
                    if (method >= 3) cf::dot4_asm(tile.data()+alignment,x.data(),blocks,out,cf::AsmVariant(method-3));
                    const Errors errors = check(expected,out);
                    worst.l2 = std::max(worst.l2,errors.l2);
                    worst.max_normalized = std::max(worst.max_normalized,errors.max_normalized);
                    ++cases;
                }
                if (alignment == 1 && blocks == 88 && mode == 6) {
                    std::atomic<bool> passed{true};
                    std::vector<std::thread> threads;
                    for (size_t t = 0; t < 8; ++t) {
                        threads.emplace_back([&] {
                            try {
                                for (size_t repetition = 0; repetition < 20; ++repetition) {
                                    float local[4];
                                    cf::dot4_arm(tile.data()+alignment,x.data(),blocks,local);
                                    check(expected,local);
                                    cf::dot4_lut(planes.data()+alignment,x.data(),lut.data(),blocks,local);
                                    check(expected,local);
                                    for (const auto variant : {cf::AsmVariant::Baseline,cf::AsmVariant::Unroll2,cf::AsmVariant::Prefetch512}) {
                                        cf::dot4_asm(tile.data()+alignment,x.data(),blocks,local,variant);
                                        check(expected,local);
                                    }
                                }
                            } catch (...) { passed = false; }
                        });
                    }
                    for (auto & thread : threads) thread.join();
                    require(passed, "independent multithreaded outputs");
                }
            }
        }
    }
    std::cout << "{\"test\":\"kernels\",\"status\":\"pass\",\"cases\":" << cases
              << ",\"max_relative_l2\":" << worst.l2
              << ",\"max_abs_normalized\":" << worst.max_normalized
              << ",\"relative_l2_limit\":" << kRelativeL2
              << ",\"max_abs_normalized_limit\":" << kMaximumAbsolute
              << ",\"arm_dotprod_executed\":" << (cf::has_dotprod()?"true":"false")
              << ",\"assembly_executed\":" << (cf::has_asm()?"true":"false")
              << ",\"lut_implementation\":\"portable-int16-subsets\""
              << ",\"oracle\":\"independent-double-original-q4-times-same-q8-codes\"}\n";
}

#if defined(__linux__)
class GuardedRegion {
public:
    GuardedRegion(size_t bytes, bool at_end) {
        const long queried = sysconf(_SC_PAGESIZE);
        require(queried>0,"cannot determine page size");
        const size_t page = size_t(queried);
        const size_t usable = ((std::max(bytes,size_t(1))+page-1)/page)*page;
        mapped_bytes_ = usable+2*page;
        base_ = mmap(nullptr,mapped_bytes_,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        require(base_!=MAP_FAILED,"guarded mmap failed");
        auto * beginning = static_cast<uint8_t *>(base_)+page;
        if (bytes && mprotect(beginning,usable,PROT_READ|PROT_WRITE)) {
            munmap(base_,mapped_bytes_);
            base_=MAP_FAILED;
            throw std::runtime_error("guarded mprotect failed");
        }
        data_ = at_end ? beginning+usable-bytes : beginning;
    }
    ~GuardedRegion() { if(base_!=MAP_FAILED)munmap(base_,mapped_bytes_); }
    GuardedRegion(const GuardedRegion &) = delete;
    GuardedRegion & operator=(const GuardedRegion &) = delete;
    uint8_t * data() { return data_; }
private:
    void * base_ = MAP_FAILED;
    size_t mapped_bytes_ = 0;
    uint8_t * data_ = nullptr;
};
#endif

void guards_and_abi() {
#if defined(__linux__)
    uint32_t state=0x1a2b3c4du;
    size_t cases=0,abi_cases=0;
    for (const size_t blocks : {size_t(0),size_t(1),size_t(2),size_t(3),size_t(7),size_t(8),size_t(9),size_t(21),size_t(22),size_t(23),size_t(87),size_t(88),size_t(89)}) {
        std::vector<uint8_t> original(blocks*72),tile(blocks*72),planes(blocks*72);
        std::vector<cf::Q8Block> x(blocks);
        for(size_t b=0;b<blocks;++b) {
            x[b].d=uint16_t(0x2400+(random_code(state)%1024));
            for(auto & code:x[b].qs)code=int8_t(int(random_code(state)>>24)-128);
            for(size_t row=0;row<4;++row) {
                auto * raw=original.data()+(row*blocks+b)*18;
                put16(raw,uint16_t(0x2800+(random_code(state)%1024)));
                for(size_t j=0;j<16;++j)raw[2+j]=uint8_t(random_code(state)>>24);
            }
        }
        pack_original(original,blocks,tile.data(),planes.data());
        const auto expected=reference(original,x);
        // Independent placement of each range against its lower/upper guard
        // exercises all eight combinations, including null-length inputs.
        for(unsigned positions=0;positions<8;++positions) {
            GuardedRegion weights(tile.size(),positions&1);
            GuardedRegion activations(x.size()*sizeof(cf::Q8Block),positions&2);
            GuardedRegion outputs(4*sizeof(float),positions&4);
            if(blocks) {
                std::memcpy(weights.data(),tile.data(),tile.size());
                std::memcpy(activations.data(),x.data(),x.size()*sizeof(cf::Q8Block));
            }
            auto * q8=reinterpret_cast<const cf::Q8Block *>(activations.data());
            auto * out=reinterpret_cast<float *>(outputs.data());
            cf::dot4_portable(weights.data(),q8,blocks,out);check(expected,out);++cases;
            cf::dot4_arm(weights.data(),q8,blocks,out);check(expected,out);++cases;
            for(const auto variant:{cf::AsmVariant::Baseline,cf::AsmVariant::Unroll2,cf::AsmVariant::Prefetch512}) {
                cf::dot4_asm(weights.data(),q8,blocks,out,variant);check(expected,out);++cases;
#if defined(CF_ENABLE_ASM) && CF_ENABLE_ASM && defined(__aarch64__)
                if(const auto kernel=cf::resolve_asm(variant)) {
                    require(cf_q4_tile4_a76_abi_probe(weights.data(),q8,blocks,out,kernel)==0,"assembly clobbered callee-preserved registers");
                    check(expected,out);
                    ++abi_cases;
                }
#endif
            }
        }
    }
    std::cout << "{\"test\":\"guard-pages-and-abi\",\"status\":\"pass\",\"guard_cases\":" << cases
              << ",\"abi_cases\":" << abi_cases << ",\"assembly_executed\":" << (cf::has_asm()?"true":"false")
              << ",\"guarded_ranges\":\"separate-weights-q8-output-both-boundaries\"}\n";
#else
    std::cout << "{\"test\":\"guard-pages-and-abi\",\"status\":\"not-run\",\"reason\":\"requires-Linux-mmap\"}\n";
#endif
}

using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}

void convert_planes(const std::vector<uint8_t> & tile, std::vector<uint8_t> & planes) {
    for (size_t b = 0; b < tile.size()/72; ++b) {
        std::memcpy(planes.data()+b*72,tile.data()+b*72,8);
        std::memset(planes.data()+b*72+8,0,64);
        for (size_t row = 0; row < 4; ++row) {
            const uint8_t * raw = tile.data()+b*72+8+row*16;
            for (size_t j = 0; j < 32; ++j) {
                const uint8_t code = (raw[j%16] >> (j<16?0:4)) & 15;
                for (size_t bit = 0; bit < 4; ++bit)
                    planes[b*72+8+row*16+(j/4)*2+bit/2] |= uint8_t(((code>>bit)&1)<<(j%4+(bit%2)*4));
            }
        }
    }
}

void benchmark(size_t k, size_t m, size_t experts, bool distinct_inputs, size_t iterations) {
    require(k%32 == 0 && m%4 == 0,"benchmark shape must satisfy packed contract");
    const size_t blocks = k/32, tiles = m/4;
    const std::vector<size_t> selected = experts == 1 ? std::vector<size_t>{0}
        : std::vector<size_t>{0,13,29,47,63,81,103,127};
    const size_t input_count = distinct_inputs ? selected.size() : 1;
    std::vector<uint8_t> packed(experts*tiles*blocks*72);
    std::vector<uint8_t> planes(packed.size());
    std::vector<cf::Q8Block> x(input_count*blocks);
    std::vector<int16_t> lut(input_count*blocks*128);
    std::vector<float> output(selected.size()*m);
    uint32_t state = 0xcafef00du;
    for (size_t b = 0; b < packed.size()/72; ++b) {
        for (size_t row = 0; row < 4; ++row) put16(packed.data()+b*72+row*2,uint16_t(0x2800+(random_code(state)%1024)));
        for (size_t j = 8; j < 72; ++j) packed[b*72+j] = uint8_t(random_code(state)>>24);
    }
    for (auto & block : x) {
        block.d = uint16_t(0x2000+(random_code(state)%1024));
        for (auto & q : block.qs) q = int8_t(int(random_code(state)>>24)-128);
    }
    const auto conversion_start = Clock::now();
    convert_planes(packed,planes);
    const double conversion_ms = elapsed_ms(conversion_start);
    for (size_t method = 0; method < 6; ++method) {
        // Cache dispatch outside the hot tile loop, as the integration adapter
        // should do. The wrapper fallback remains tested above on every host.
        const auto asm_kernel=method>=3?cf::resolve_asm(cf::AsmVariant(method-3)):nullptr;
        std::vector<double> prep_times, multiply_times;
        for (size_t iteration = 0; iteration <= iterations; ++iteration) {
            auto start = Clock::now();
            if (method == 2) cf::build_lut(x.data(),x.size(),lut.data());
            const double prep_ms = elapsed_ms(start);
            start = Clock::now();
            for (size_t s = 0; s < selected.size(); ++s) {
                const size_t input = distinct_inputs ? s : 0;
                for (size_t tile = 0; tile < tiles; ++tile) {
                    const size_t offset = (selected[s]*tiles + tile)*blocks*72;
                    const auto * activation = x.data()+input*blocks;
                    float * result = output.data()+s*m+tile*4;
                    if (method == 0) cf::dot4_portable(packed.data()+offset,activation,blocks,result);
                    if (method == 1) cf::dot4_arm(packed.data()+offset,activation,blocks,result);
                    if (method == 2) cf::dot4_lut(planes.data()+offset,activation,lut.data()+input*blocks*128,blocks,result);
                    if (method >= 3) {
                        if(asm_kernel)asm_kernel(packed.data()+offset,activation,blocks,result);
                        else cf::dot4_portable(packed.data()+offset,activation,blocks,result);
                    }
                }
            }
            const double multiply_ms = elapsed_ms(start);
            if (iteration) { prep_times.push_back(prep_ms); multiply_times.push_back(multiply_ms); }
        }
        double prep_mean = 0, multiply_mean = 0, checksum = 0;
        for (double t : prep_times) prep_mean += t/iterations;
        for (double t : multiply_times) multiply_mean += t/iterations;
        for (float v : output) checksum += v;
        const char * labels[]={"portable",cf::has_dotprod()?"arm-dotprod":"portable-arm-request-fallback","portable-bitplane-lut",
            cf::has_asm()?"asm-a76":"portable-asm-request-fallback",
            cf::has_asm()?"asm-a76-unroll2":"portable-asm-unroll2-request-fallback",
            cf::has_asm()?"asm-a76-prefetch512":"portable-asm-prefetch512-request-fallback"};
        const char * label=labels[method];
        std::cout << "{\"benchmark\":\"synthetic-model-shape\",\"K\":" << k << ",\"M\":" << m
                  << ",\"E\":" << experts << ",\"selected_experts\":" << selected.size()
                  << ",\"kernel\":\"" << label << "\",\"iterations\":" << iterations
                  << ",\"threads\":1,\"unique_inputs\":" << input_count
                  << ",\"q8_preparation\":\"excluded-common-already-quantized-input\""
                  << ",\"lut_preparation_mean_ms\":" << (method==2?prep_mean:0)
                  << ",\"multiplication_mean_ms\":" << multiply_mean
                  << ",\"multiplication_min_ms\":" << *std::min_element(multiply_times.begin(),multiply_times.end())
                  << ",\"multiplication_max_ms\":" << *std::max_element(multiply_times.begin(),multiply_times.end())
                  << ",\"total_mean_ms\":" << multiply_mean+(method==2?prep_mean:0)
                  << ",\"offline_full_tensor_bitplane_conversion_ms\":" << conversion_ms
                  << ",\"one_format_full_weight_bytes\":" << packed.size()
                  << ",\"harness_both_formats_weight_bytes\":" << packed.size()+planes.size()
                  << ",\"selected_logical_weight_bytes\":" << selected.size()*tiles*blocks*72
                  << ",\"q8_input_bytes\":" << x.size()*sizeof(cf::Q8Block)
                  << ",\"lut_scratch_bytes\":" << (method==2?lut.size()*sizeof(int16_t):0)
                  << ",\"output_bytes\":" << output.size()*sizeof(float)
                  << ",\"working_set\":\"" << (experts==1?"cache-sized":"larger-than-Pi-L3")
                  << "\",\"dram_bytes_measured\":false,\"checksum\":" << checksum << "}\n";
    }
}
} // namespace

int main(int argc, char ** argv) {
    try {
        bool bench = false;
        size_t iterations = 5;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--bench") bench = true;
            else if (arg == "--iterations" && i+1 < argc) {
                const std::string value(argv[++i]);
                size_t end = 0;
                iterations = std::stoul(value,&end);
                require(end==value.size() && iterations>=1 && iterations<=1000,"iterations must be 1..1000");
            } else throw std::runtime_error("usage: kernels_test [--bench] [--iterations 1..1000]");
        }
        std::cout << std::setprecision(10);
        correctness();
        guards_and_abi();
        if (bench) {
            benchmark(704,128,1,false,iterations);
            benchmark(2816,1408,128,false,iterations);
            benchmark(704,2816,128,true,iterations);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "{\"status\":\"fail\",\"error\":\"" << error.what() << "\"}\n";
        return 1;
    }
}
