// Real captured Gemma4 gate/up operator comparison. No model/runtime simulation.
#include "kernels.h"
#include "kleidi_compare.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "repack.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>

extern "C" void cf_sha256_hash(unsigned char *, const unsigned char *, size_t);
using json = nlohmann::ordered_json;
namespace {
using Clock = std::chrono::steady_clock;
constexpr double kRelativeL2 = 2e-5, kNormalizedAbsolute = 2e-4;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
void require(bool ok,const std::string & why) {if(!ok)throw std::runtime_error(why);}
std::string digest(const void * p,size_t size) {
    unsigned char hash[32];cf_sha256_hash(hash,static_cast<const unsigned char *>(p),size);
    static const char hex[]="0123456789abcdef";std::string out(64,'0');
    for(size_t i=0;i<32;++i){out[i*2]=hex[hash[i]>>4];out[i*2+1]=hex[hash[i]&15];}return out;
}
json read_json(const std::filesystem::path & p) {
    std::ifstream f(p);require(bool(f),"cannot open "+p.string());json j;f>>j;return j;
}
void read_exact(std::ifstream & f,void * p,size_t size) {
    f.read(static_cast<char *>(p),static_cast<std::streamsize>(size));
    require(size_t(f.gcount())==size,"short binary read");
}
double temperature() {
    std::ifstream f("/sys/class/thermal/thermal_zone0/temp");double value=-1000;
    if(f>>value)return value/1000;return -1;
}
json proc_memory() {
    rusage r{};getrusage(RUSAGE_SELF,&r);json j={{"peak_rss_kib",r.ru_maxrss},{"major_faults",r.ru_majflt}};
    for(const char * name:{"status","io","smaps_rollup"}) {
        std::ifstream f(std::string("/proc/self/")+name);
        j[name]=std::string(std::istreambuf_iterator<char>(f),{});
    }return j;
}

// Fixed worker pool: no thread creation, task allocation, or nested parallelism
// during iterations. Every method receives the same contiguous four-row jobs.
class Pool {
public:
    using Function=void(*)(void *,size_t,size_t);
    explicit Pool(size_t workers):workers_(workers) {
        threads_.reserve(workers);
        for(size_t i=0;i<workers;++i)threads_.emplace_back([this,i]{worker(i);});
    }
    ~Pool() {
        {std::lock_guard<std::mutex> lock(mutex_);stop_=true;}start_.notify_all();
        for(auto & t:threads_)t.join();
    }
    void run(Function fn,void * state,size_t jobs) {
        std::unique_lock<std::mutex> lock(mutex_);
        fn_=fn;state_=state;jobs_=jobs;pending_=workers_;error_=nullptr;++generation_;
        start_.notify_all();done_.wait(lock,[this]{return pending_==0;});
        if(error_)std::rethrow_exception(error_);
    }
private:
    void worker(size_t index) {
        size_t seen=0;
        while(true) {
            std::unique_lock<std::mutex> lock(mutex_);
            start_.wait(lock,[&]{return stop_||generation_!=seen;});if(stop_)return;
            seen=generation_;const auto fn=fn_;auto state=state_;const size_t jobs=jobs_;
            lock.unlock();std::exception_ptr failure;
            try{fn(state,jobs*index/workers_,jobs*(index+1)/workers_);}catch(...){failure=std::current_exception();}
            lock.lock();if(failure&&!error_)error_=failure;if(--pending_==0)done_.notify_one();
        }
    }
    size_t workers_,jobs_=0,pending_=0,generation_=0;
    bool stop_=false;Function fn_=nullptr;void * state_=nullptr;
    std::mutex mutex_;std::condition_variable start_,done_;
    std::exception_ptr error_;std::vector<std::thread> threads_;
};

struct Repacked {
    ggml_context * context=nullptr;ggml_backend_buffer_t buffer=nullptr;ggml_tensor * tensor=nullptr;
    double allocation_and_pack_ms=0;size_t allocated_bytes=0;
    ~Repacked(){if(buffer)ggml_backend_buffer_free(buffer);if(context)ggml_free(context);}
    Repacked(const Repacked &)=delete;Repacked & operator=(const Repacked &)=delete;
    Repacked(const uint8_t * original,size_t k,size_t m) {
        const auto begin=Clock::now();
        ggml_init_params params{};params.mem_size=ggml_tensor_overhead()+2048;params.no_alloc=true;
        context=ggml_init(params);require(context,"reference context allocation failed");
        tensor=ggml_new_tensor_2d(context,GGML_TYPE_Q4_0,k,m);
        ggml_set_name(tensor,"operator_bench_actual_q4_expert");
        buffer=ggml_backend_alloc_ctx_tensors_from_buft(context,ggml_backend_cpu_repack_buffer_type());
        require(buffer&&tensor->data&&tensor->extra,"actual CPU_REPACK allocation/dispatch failed");
        ggml_backend_tensor_set(tensor,original,0,ggml_nbytes(tensor));
        allocated_bytes=ggml_backend_buffer_get_size(buffer);allocation_and_pack_ms=elapsed(begin);
        // Verify the actually produced 4x4 layout before calling its direct GEMV.
        const size_t blocks=k/32,row_bytes=blocks*18;
        const auto packed=static_cast<const uint8_t *>(tensor->data);
        for(size_t row=0;row<m;++row)for(size_t b=0;b<blocks;++b) {
            const auto src=original+row*row_bytes+b*18;
            const auto tile=packed+((row/4)*blocks+b)*72;
            require(std::memcmp(src,tile+(row%4)*2,2)==0,"CPU_REPACK scale roundtrip mismatch");
            for(size_t q=0;q<16;++q)
                require(uint8_t(tile[8+(q/4)*16+(row%4)*4+q%4]^0x88)==src[q+2],"CPU_REPACK is not verified q4_0_4x4 layout");
        }
    }
};

enum class Method { Reference, Repack, RepackRange, Portable, Neon, Asm, Kleidi, KleidiRange };
struct Candidate {const char * name;Method method;cf::AsmKernelFn assembly=nullptr;};
struct Expert {
    int id=0;std::vector<uint8_t> tile,original;
    std::unique_ptr<Repacked> repack;
    cf::kleidi::Weights kleidi;
};
struct State {
    size_t k=0,m=0,blocks=0,tiles=0;
    std::vector<Expert> experts;
    std::vector<float> input,output,reference;
    std::vector<cf::Q8Block> q8;
    cf::kleidi::Activation kleidi_input,native_input;
    ggml_vec_dot_t vec_dot=nullptr;
    Candidate selected{};
};
void rows(void * context,size_t first,size_t end) {
    auto & s=*static_cast<State *>(context);
    const size_t row_bytes=s.blocks*18;
    // The same worker partition also permits the external APIs' natural range
    // calls. Coalesce only within one expert; routing/input dependencies remain
    // unchanged. Keep tile4 candidates below to distinguish call overhead.
    if(s.selected.method==Method::RepackRange||s.selected.method==Method::KleidiRange) {
        for(size_t job=first;job<end;) {
            const size_t e=job/s.tiles,tile=job%s.tiles,row=tile*4;
            const size_t stop=std::min(end,(e+1)*s.tiles),count=(stop-job)*4;
            auto & w=s.experts[e];
            if(s.selected.method==Method::RepackRange) {
                ggml_gemv_q4_0_4x4_q8_0(int(s.k),s.output.data()+e*s.m+row,count,
                    static_cast<uint8_t *>(w.repack->tensor->data)+tile*s.blocks*72,
                    s.q8.data(),1,int(count));
            } else {
                cf::kleidi::run(w.kleidi,s.kleidi_input,s.output.data()+e*s.m,row,count);
            }
            job=stop;
        }
        return;
    }
    for(size_t job=first;job<end;++job) {
        const size_t e=job/s.tiles,tile=job%s.tiles,row=tile*4;
        auto & w=s.experts[e];float * out=s.output.data()+e*s.m+row;
        const uint8_t * packed=w.tile.data()+tile*s.blocks*72;
        switch(s.selected.method) {
        case Method::Reference:
            for(size_t r=0;r<4;++r)s.vec_dot(int(s.k),out+r,0,w.original.data()+(row+r)*row_bytes,0,s.q8.data(),0,1);
            break;
        case Method::Repack:
            ggml_gemv_q4_0_4x4_q8_0(int(s.k),out,4,static_cast<uint8_t *>(w.repack->tensor->data)+tile*s.blocks*72,s.q8.data(),1,4);
            break;
        case Method::Portable:cf::dot4_portable(packed,s.q8.data(),s.blocks,out);break;
        case Method::Neon:cf::dot4_arm(packed,s.q8.data(),s.blocks,out);break;
        case Method::Asm:s.selected.assembly(packed,s.q8.data(),s.blocks,out);break;
        case Method::Kleidi:cf::kleidi::run(w.kleidi,s.kleidi_input,s.output.data()+e*s.m,row,4);break;
        case Method::RepackRange:case Method::KleidiRange:throw std::logic_error("range method reached tile4 loop");
        }
    }
}
json compare(const std::vector<float> & reference,const std::vector<float> & candidate) {
    require(reference.size()==candidate.size(),"comparison shape mismatch");
    double norm=0,error=0,maxdiff=0,maxref=0;size_t nonfinite=0,bit_mismatches=0;
    for(size_t i=0;i<reference.size();++i) {
        if(std::memcmp(&reference[i],&candidate[i],sizeof(float)))++bit_mismatches;
        const double a=reference[i],b=candidate[i];if(!std::isfinite(a)||!std::isfinite(b)){++nonfinite;continue;}
        norm+=a*a;error+=(a-b)*(a-b);maxdiff=std::max(maxdiff,std::abs(a-b));maxref=std::max(maxref,std::abs(a));
    }
    const double relative=std::sqrt(error/std::max(norm,1e-30)),scaled=maxdiff/std::max(1.0,maxref);
    return {{"values",reference.size()},{"nonfinite",nonfinite},{"relative_l2",relative},{"max_absolute",maxdiff},
            {"max_abs_normalized",scaled},{"relative_l2_limit",kRelativeL2},{"max_abs_normalized_limit",kNormalizedAbsolute},
            {"bit_mismatches",bit_mismatches},{"bit_identical",bit_mismatches==0},
            {"passed",!nonfinite&&relative<=kRelativeL2&&scaled<=kNormalizedAbsolute}};
}
void enforce_cf_bit_identity(json & check,const Candidate & method) {
    const bool required=cf::has_dotprod()&&(method.method==Method::Portable||method.method==Method::Neon||method.method==Method::Asm);
    check["cf_reference_bit_identity_required"]=required;
    if(required&&!check.at("bit_identical").get<bool>())check["passed"]=false;
}
json native_code_difference(const State & s) {
    const auto * native=reinterpret_cast<const uint8_t *>(s.native_input.packed.data());
    size_t scales=0,codes=0;int maximum=0;
    for(size_t b=0;b<s.blocks;++b) {
        uint16_t d;std::memcpy(&d,native+b*34,2);if(d!=s.q8[b].d)++scales;
        for(size_t j=0;j<32;++j) {
            const int diff=int(static_cast<int8_t>(native[b*34+2+j]))-int(s.q8[b].qs[j]);
            if(diff)++codes;maximum=std::max(maximum,std::abs(diff));
        }
    }
    return {{"scale_bit_mismatches",scales},{"integer_code_mismatches",codes},{"maximum_integer_code_delta",maximum},
            {"same_reference_codes",!scales&&!codes}};
}

State load_case(const json & tensor,const std::filesystem::path & payload,const json & capture,
                const std::vector<float> & input,bool small,bool do_repack,bool do_kleidi,json & preparation) {
    State s;s.k=capture.at("K");s.m=small?16:capture.at("M").get<size_t>();s.blocks=s.k/32;s.tiles=s.m/4;
    s.input=input;s.q8.resize(s.blocks);s.vec_dot=ggml_get_type_traits_cpu(GGML_TYPE_Q4_0)->vec_dot;
    require(s.vec_dot&&ggml_get_type_traits_cpu(GGML_TYPE_Q4_0)->vec_dot_type==GGML_TYPE_Q8_0,"reference Q4_0 dot path incompatible");
    const auto ids=capture.at("ids").get<std::vector<int>>();
    const size_t selected=small?1:ids.size(),full_expert_bytes=capture.at("M").get<size_t>()*s.blocks*18;
    s.experts.reserve(selected);preparation=json::array();std::ifstream f(payload,std::ios::binary);require(bool(f),"cannot open packed payload");
    for(size_t i=0;i<selected;++i) {
        Expert e;e.id=ids[i];const size_t count=s.m*s.blocks*18;
        e.tile.resize(count);f.seekg(static_cast<std::streamoff>(size_t(e.id)*full_expert_bytes));read_exact(f,e.tile.data(),count);
        const auto start=Clock::now();e.original.resize(count);
        for(size_t row=0;row<s.m;++row)for(size_t b=0;b<s.blocks;++b) {
            const auto packed=e.tile.data()+((row/4)*s.blocks+b)*72;
            auto raw=e.original.data()+(row*s.blocks+b)*18;
            std::memcpy(raw,packed+(row%4)*2,2);std::memcpy(raw+2,packed+8+(row%4)*16,16);
            uint16_t scale;std::memcpy(&scale,raw,2);require((scale&0x7c00)!=0x7c00,"nonfinite source weight scale");
        }
        json prep={{"expert_id",e.id},{"cf_packed_bytes",count},{"compact_original_bytes",count},
                   {"inverse_layout_ms",elapsed(start)},{"selected_cf_bytes_sha256",digest(e.tile.data(),count)},
                   {"selected_original_bytes_sha256",digest(e.original.data(),count)}};
        if(do_repack) {
            e.repack=std::make_unique<Repacked>(e.original.data(),s.k,s.m);
            prep["upstream_repack_allocation_and_pack_ms"]=e.repack->allocation_and_pack_ms;
            prep["upstream_repack_buffer_bytes"]=e.repack->allocated_bytes;prep["upstream_4x4_roundtrip"]=true;
        }
        if(do_kleidi) {
            const auto begin=Clock::now();e.kleidi=cf::kleidi::prepare(e.original.data(),s.k,s.m);
            prep["kleidi_prepare_total_ms"]=elapsed(begin);prep["kleidi_pack_ms"]=e.kleidi.pack_ms;
            prep["kleidi_validation_ms"]=e.kleidi.validation_ms;prep["kleidi_packed_bytes"]=e.kleidi.packed.size();
            prep["kleidi_roundtrip"]=e.kleidi.roundtrip;
        }
        preparation.push_back(std::move(prep));s.experts.push_back(std::move(e));
    }
    s.output.resize(selected*s.m);s.reference.resize(selected*s.m);
    if(do_kleidi){s.kleidi_input=cf::kleidi::prepare_activation(s.k);s.native_input=cf::kleidi::prepare_activation(s.k);}
    (void)tensor;return s;
}
json accounted(const State & s) {
    size_t cf_bytes=0,original=0,repack=0,klei=0;
    for(const auto & e:s.experts){cf_bytes+=e.tile.capacity();original+=e.original.capacity();
        if(e.repack)repack+=e.repack->allocated_bytes;klei+=e.kleidi.packed.capacity();}
    const size_t activation=s.input.capacity()*4+s.q8.capacity()*sizeof(cf::Q8Block)+s.kleidi_input.packed.capacity()+s.native_input.packed.capacity();
    const size_t outputs=(s.output.capacity()+s.reference.capacity())*4;
    return {{"cf_packed_capacity_bytes",cf_bytes},{"original_compact_capacity_bytes",original},{"upstream_repack_buffer_bytes",repack},
            {"kleidi_packed_capacity_bytes",klei},{"activation_capacity_bytes",activation},{"output_reference_capacity_bytes",outputs},
            {"counted_live_buffer_bytes",cf_bytes+original+repack+klei+activation+outputs},
            {"not_in_buffer_sum","allocator metadata, thread stacks, runtime libraries, source file page cache, JSON metadata; see process RSS"}};
}
} // namespace

int main(int argc,char ** argv) {
    std::ofstream output;
    try {
        std::filesystem::path plan_path,capture_path,out_path;int iterations=21;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];require(i+1<argc,"every argument requires a value");const std::string value=argv[++i];
            if(arg=="--plan")plan_path=value;else if(arg=="--capture")capture_path=value;
            else if(arg=="--out")out_path=value;else if(arg=="--iterations")iterations=std::stoi(value);else throw std::runtime_error("unknown option "+arg);
        }
        require(!plan_path.empty()&&!capture_path.empty()&&!out_path.empty(),"required: --plan --capture --out [--iterations 21]");
        require(iterations>=1&&iterations<=10000,"invalid iteration count");
        require(!std::filesystem::exists(out_path),"output already exists; use a new result path");
        if(!out_path.parent_path().empty())std::filesystem::create_directories(out_path.parent_path());
        output.open(out_path);require(bool(output),"cannot create output");
        auto emit=[&](const json & record){output<<record.dump()<<'\n';output.flush();require(bool(output),"output write failed");};
        const auto plan=read_json(plan_path),capture=read_json(capture_path);
        require(plan.at("format")=="cf-engine-plan"&&plan.at("version")==1,"unsupported plan");
        require(capture.at("K")==2816&&capture.at("M")==1408&&capture.at("E")==128&&capture.at("input_vectors")==1,
                "requires actual Gemma4 gate/up K2816 M1408 E128 single shared activation");
        const auto ids=capture.at("ids").get<std::vector<int>>();std::set<int> unique(ids.begin(),ids.end());
        require(ids.size()==8&&unique.size()==8&&*unique.begin()>=0&&*unique.rbegin()<128,"capture must contain eight distinct valid expert IDs");
        const std::string weight_name=capture.at("weight_name");require(weight_name.find("ffn_gate_up_exps.weight")!=std::string::npos,"capture is not gate/up");
        json tensor;size_t matches=0;for(const auto & t:plan.at("tensors"))if(t.at("name")==weight_name){tensor=t;++matches;}
        require(matches==1&&tensor.at("type")=="Q4_0"&&tensor.at("layout")=="q4-tile4-v1","expected unique Q4_0 tile4 gate/up tensor");
        require(tensor.at("shape")==json::array({2816,1408,128})&&tensor.at("bytes")==285474816,"invalid tensor bounds");
        const auto base=std::filesystem::canonical(std::filesystem::absolute(plan_path).parent_path());
        const std::filesystem::path relative=tensor.at("file").get<std::string>();require(!relative.is_absolute(),"absolute payload path rejected");
        const auto payload=std::filesystem::canonical(base/relative),rel=payload.lexically_relative(base);
        require(!rel.empty()&&*rel.begin()!=".."&&std::filesystem::file_size(payload)==285474816,"payload bounds/path mismatch");
        auto floats=capture_path;floats.replace_extension(".f32");require(std::filesystem::file_size(floats)==2816*4,"capture f32 size mismatch");
        std::vector<float> input(2816);std::ifstream f(floats,std::ios::binary);read_exact(f,input.data(),input.size()*4);
        for(float v:input)require(std::isfinite(v),"nonfinite captured activation");
        const uint16_t endian=1;require(*reinterpret_cast<const uint8_t *>(&endian)==1,"little-endian inputs required");
        ggml_cpu_init();const auto quantize=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;require(quantize,"reference Q8 quantizer missing");
        const bool repack=cf::has_dotprod()&&ggml_cpu_has_neon()&&ggml_cpu_has_dotprod()&&!ggml_cpu_has_matmul_int8()&&!ggml_cpu_has_avx2();
        const bool kleidi=cf::kleidi::available();
        std::vector<Candidate> methods{{"upstream_vec_dot",Method::Reference},{"cf_portable",Method::Portable}};
        if(repack) {
            methods.push_back({"upstream_cpu_repack_4x4",Method::Repack});
            methods.push_back({"upstream_cpu_repack_range",Method::RepackRange});
        }
        if(cf::has_dotprod())methods.push_back({"cf_neon",Method::Neon});
        const cf::AsmVariant variants[]={cf::AsmVariant::Baseline,cf::AsmVariant::Unroll2,cf::AsmVariant::Prefetch512};
        const char * names[]={"cf_asm_baseline","cf_asm_unroll2","cf_asm_prefetch512"};
        for(size_t i=0;i<3;++i)if(auto fn=cf::resolve_asm(variants[i]))methods.push_back({names[i],Method::Asm,fn});
        if(kleidi) {
            methods.push_back({"kleidiai_same_q8",Method::Kleidi});
            methods.push_back({"kleidiai_same_q8_range",Method::KleidiRange});
        }
        json method_names=json::array();for(auto & method:methods)method_names.push_back(method.name);
        emit({{"record","metadata"},{"format","cf-operator-bench-v1"},{"plan",plan_path.string()},{"capture",capture},
              {"activation_sha256",digest(input.data(),input.size()*4)},{"tensor",tensor},{"iterations",iterations},
              {"kernels_executed",method_names},{"upstream_repack_4x4_available",repack},{"kleidi_available",kleidi},{"asm_available",cf::has_asm()},
              {"artifact_identity_note","Only selected expert ranges are read; full payload SHA256 is declared from previously verified plan, not rehashed here. Selected range hashes are measured."},
              {"execution_contract","same-Q4_0-weights-and-reference-Q8-codes operator comparison; native Kleidi quantization separately labeled"},
              {"schedule","persistent workers, contiguous four-row job ranges, all methods same scheduler; no nested parallelism"},
              {"range_candidates","Same worker partition; upstream/Kleidi range APIs called once per contiguous rows within each expert. Other candidates retain tile4 calls."},
              {"order","rotate by repetition, reverse odd repetitions; one unmeasured warmup per method/thread count"},
              {"traffic_note","working-set bytes are logical storage sizes, not measured DRAM traffic; no global cache manipulation"},
              {"limits",{{"relative_l2",kRelativeL2},{"max_abs_normalized",kNormalizedAbsolute}}}});
        bool all_passed=true;
        for(bool small:{true,false}) {
            json preparations;State s=load_case(tensor,payload,capture,input,small,repack,kleidi,preparations);
            const char * label=small?"cache-sized-M16-E1":"selected-experts-M1408-E8";
            quantize(s.input.data(),s.q8.data(),int64_t(s.k));if(kleidi)cf::kleidi::pack_q8(s.kleidi_input,s.q8.data());
            s.selected=methods.front();rows(&s,0,s.experts.size()*s.tiles);s.reference=s.output;
            emit({{"record","case"},{"case",label},{"K",s.k},{"M",s.m},{"selected_experts",s.experts.size()},
                  {"weights_per_method_bytes",s.m*s.blocks*18*s.experts.size()},
                  {"exceeds_reported_Pi_L3",s.m*s.blocks*18*s.experts.size()>2*1024*1024},
                  {"preparation",preparations},{"memory",accounted(s)},{"process_memory",proc_memory()}});
            for(size_t workers:{size_t(1),size_t(2),size_t(4)}) {
                Pool pool(workers);const size_t jobs=s.experts.size()*s.tiles;
                for(auto method:methods){s.selected=method;pool.run(rows,&s,jobs);auto check=compare(s.reference,s.output);
                    enforce_cf_bit_identity(check,method);
                    all_passed=all_passed&&check["passed"].get<bool>();
                    emit({{"record","correctness"},{"case",label},{"workers",workers},{"kernel",method.name},{"numerical",check}});}
                std::vector<size_t> order(methods.size());std::iota(order.begin(),order.end(),0);
                const double before=temperature();
                for(int iteration=0;iteration<iterations;++iteration) {
                    const auto prep_begin=Clock::now();quantize(s.input.data(),s.q8.data(),int64_t(s.k));const double q8_ms=elapsed(prep_begin);
                    double copy_ms=0;
                    if(kleidi) {
                        const auto copy_begin=Clock::now();cf::kleidi::pack_q8(s.kleidi_input,s.q8.data());copy_ms=elapsed(copy_begin);
                        const auto native_begin=Clock::now();cf::kleidi::pack_activation(s.native_input,s.input.data());const double native_ms=elapsed(native_begin);
                        emit({{"record","native-kleidi-preparation"},{"case",label},{"workers",workers},{"iteration",iteration},
                              {"native_quant_pack_ms",native_ms},{"reference_quant_ms",q8_ms},{"same_code_copy_ms",copy_ms},
                              {"code_difference",native_code_difference(s)},{"note","Native quantized codes not used in same-code kernel timings."}});
                    }
                    std::iota(order.begin(),order.end(),0);std::rotate(order.begin(),order.begin()+iteration%order.size(),order.end());
                    if(iteration%2)std::reverse(order.begin(),order.end());
                    for(size_t position=0;position<order.size();++position) {
                        s.selected=methods[order[position]];const auto begin=Clock::now();pool.run(rows,&s,jobs);const double kernel_ms=elapsed(begin);
                        auto check=compare(s.reference,s.output);enforce_cf_bit_identity(check,s.selected);all_passed=all_passed&&check["passed"].get<bool>();
                        const double extra=(s.selected.method==Method::Kleidi||s.selected.method==Method::KleidiRange)?copy_ms:0;
                        emit({{"record","iteration"},{"case",label},{"workers",workers},{"iteration",iteration},{"order_position",position},
                              {"kernel",s.selected.name},{"reference_q8_prepare_ms",q8_ms},{"kernel_specific_prepare_ms",extra},
                              {"multiply_dispatch_sync_ms",kernel_ms},{"full_operator_ms",q8_ms+extra+kernel_ms},
                              {"numerical",check},{"checksum",std::accumulate(s.output.begin(),s.output.end(),0.0)},
                              {"preparation_note","Reference quantization measured once this repetition and charged to each candidate; Kleidi same-code copy charged additionally."}});
                    }
                }
                if(kleidi) {
                    std::swap(s.kleidi_input,s.native_input);s.selected={"kleidiai_native_q8",Method::Kleidi};pool.run(rows,&s,jobs);
                    emit({{"record","native-kleidi-output-check"},{"case",label},{"workers",workers},{"numerical",compare(s.reference,s.output)},
                          {"note","Separate quantization comparison, not the same-code correctness gate or performance result."}});
                    std::swap(s.kleidi_input,s.native_input);
                }
                emit({{"record","thread-block-end"},{"case",label},{"workers",workers},{"temperature_before_c",before},
                      {"temperature_after_c",temperature()},{"process_memory",proc_memory()}});
            }
        }
        emit({{"record","summary"},{"all_same_code_numerical_checks_passed",all_passed},{"energy_measured",false},
              {"full_model_speed_claim",false},{"note","Operator evidence only; performance candidates must still win matched end-to-end inference."}});
        std::cout<<"operator benchmark "<<(all_passed?"PASS":"FAIL")<<": "<<out_path<<'\n';return all_passed?0:1;
    } catch(const std::exception & e) {
        if(output){output<<json({{"record","error"},{"error",e.what()}}).dump()<<'\n';output.flush();}
        std::cerr<<"operator benchmark: "<<e.what()<<'\n';return 2;
    }
}
