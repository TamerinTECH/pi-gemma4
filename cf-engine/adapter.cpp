#include "adapter.h"
#include "kernels.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "quants.h"
#include "nlohmann/json.hpp"
#define sha256_init cf_sha256_init
#define sha256_update cf_sha256_update
#define sha256_final cf_sha256_final
#define sha256_hash cf_sha256_hash
extern "C" {
#include "hash/sha256/sha256.h"
}
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::ordered_json;
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double,std::milli>(b-a).count(); }
const char * env(const char * k, const char * fallback) { auto p=std::getenv(k); return p ? p : fallback; }
std::string hex(const uint8_t * p, size_t n) {
    static const char digits[]="0123456789abcdef";
    std::string r(n*2,'0'); for(size_t i=0;i<n;++i) {r[i*2]=digits[p[i]>>4];r[i*2+1]=digits[p[i]&15];} return r;
}
std::string digest(const void * p, size_t n) {
    uint8_t d[32]; sha256_hash(d,static_cast<const uint8_t *>(p),n); return hex(d,32);
}
std::string digest_file(const std::filesystem::path & p) {
    std::ifstream f(p,std::ios::binary); if(!f) throw std::runtime_error("cannot open source: "+p.string());
    std::vector<uint8_t> buf(1024*1024); sha256_t h; sha256_init(&h);
    while(f) {f.read(reinterpret_cast<char *>(buf.data()),buf.size());sha256_update(&h,buf.data(),size_t(f.gcount()));}
    if(!f.eof()) throw std::runtime_error("source read error");
    uint8_t d[32];sha256_final(&h,d);return hex(d,32);
}
struct Entry {
    std::string name, source_hash, layout;
    size_t k=0,m=0,e=0,bytes=0;
    void * data=MAP_FAILED;
    bool source_checked=false;
    bool captured=false;
    int decode_output_mod64=-1,decode_stride_mod64=-1;
};
struct Stat {
    std::string name, op, phase, dispatch, reason, weight_name;
    uint64_t calls=0, logical_weight_bytes=0;
    double elapsed_ms=0,prepare_ms=0,multiply_ms=0;
};
struct State {
    bool enabled=false,validate=false,initialized=false;
    bool request_cf=true;
    std::string kernel,report,memory="resident",phase="prefill",failure;
    std::string scheduler="contiguous";
    std::map<std::string,Entry,std::less<>> entries;
    std::unordered_map<uint64_t,Stat> stats;
    std::vector<cf::Q8Block> q8;
    std::vector<int16_t> lut;
    std::vector<float> validation;
    size_t max_vectors=512,max_blocks=88,max_output=2816*8*64;
    size_t mapped_packed=0,source_bytes=0,scratch_peak=0;
    uint64_t cf_calls=0,fallback_calls=0,validation_values=0,validation_failures=0;
    double init_ms=0,source_tensor_check_ms=0,max_rel_l2=0,max_scaled_abs=0;
    Clock::time_point start,prep_end;
    Stat * current=nullptr;
    const char * dispatch="reference";
    const char * reason="not_owned";
    double prep_ms=0,mul_ms=0;
    bool ready=false;
    FILE * logits=nullptr;
    FILE * routes=nullptr;
    std::string capture_dir;
    cf::AsmKernelFn asm_kernel=nullptr;
    std::vector<std::string> schedule;
    std::string dispatch_name="cf";
    json requests=json::array();
    double request_cf_ms=0,request_ref_ms=0,request_prepare_ms=0,request_multiply_ms=0,request_hash_start=0;
    uint64_t request_cf_calls=0,request_ref_calls=0;
    Clock::time_point ffn_start;
    bool ffn_active=false;
    double request_ffn_prefill_ms=0,request_ffn_decode_ms=0;
    uint64_t request_index=0,request_read_bytes=0;
    long request_major_faults=0;
} s;

void require(bool ok,const std::string & msg) {if(!ok)throw std::runtime_error(msg);}
uint64_t storage_read_bytes() {
    std::ifstream f("/proc/self/io");std::string key;uint64_t value;
    while(f>>key>>value)if(key=="read_bytes:")return value;
    return 0;
}
size_t checked_mul(size_t a,size_t b) {require(!a||b<=SIZE_MAX/a,"size overflow");return a*b;}
void verify_source(const json & j,const std::filesystem::path & model) {
    auto groups=json::array();groups.push_back(j.at("source").at("files"));
    if(j.at("source").contains("alternative_file_identities")&&!j.at("source").at("alternative_file_identities").empty()) {
        groups.push_back(j.at("source").at("alternative_file_identities"));
    }
    bool found=false;
    for(const auto & files:groups) {
        if(files.empty()||files[0].at("name").get<std::string>()!=model.filename().string())continue;
        for(const auto & f:files) {
            std::string name=f.at("name");
            require(std::filesystem::path(name).filename()==name,"invalid source file name");
            auto p=model.parent_path()/name;
            require(std::filesystem::file_size(p)==f.at("bytes").get<uint64_t>(),"source size mismatch: "+name);
            require(digest_file(p)==f.at("sha256").get<std::string>(),"source SHA256 mismatch: "+name);
            s.source_bytes+=f.at("bytes").get<uint64_t>();
        }
        found=true;break;
    }
    require(found,"model filename absent from source identities (supply verified split-file identity set)");
}
Entry * eligible(const ggml_tensor * dst) {
    if(!s.enabled||!s.request_cf||dst->op!=GGML_OP_MUL_MAT_ID)return nullptr;
    const auto a=dst->src[0],b=dst->src[1],ids=dst->src[2];
    auto it=s.entries.find(a->name);if(it==s.entries.end())return nullptr;
    auto & e=it->second;
    if(a->type!=GGML_TYPE_Q4_0||b->type!=GGML_TYPE_F32||ids->type!=GGML_TYPE_I32||dst->type!=GGML_TYPE_F32)return nullptr;
    if(a->ne[0]!=int64_t(e.k)||a->ne[1]!=int64_t(e.m)||a->ne[2]!=int64_t(e.e)||a->ne[3]!=1)return nullptr;
    if(!ggml_is_contiguous(a)||b->nb[0]!=4||b->ne[0]!=int64_t(e.k)||b->ne[3]!=1)return nullptr;
    if(ids->ne[0]<1||ids->ne[0]>8||ids->ne[1]<1||ids->ne[1]>64||ids->ne[2]!=1||ids->ne[3]!=1)return nullptr;
    if(b->ne[1]<1||b->ne[1]>ids->ne[0]||b->ne[2]!=ids->ne[1])return nullptr;
    if(dst->ne[0]!=int64_t(e.m)||dst->ne[1]!=ids->ne[0]||dst->ne[2]!=ids->ne[1]||dst->ne[3]!=1||dst->nb[0]!=4)return nullptr;
    if(dst->nb[1]<e.m*4||dst->nb[2]<dst->nb[1]*size_t(dst->ne[1]))return nullptr;
    if(size_t(b->ne[1]*b->ne[2])>s.max_vectors||e.k/32>s.max_blocks)return nullptr;
    return &e;
}
uint64_t stat_key(const ggml_tensor * t,const char * dispatch,const char * reason) {
    uint64_t h=1469598103934665603ull;
    for(auto p:{t->name,ggml_op_name(t->op),s.phase.c_str(),dispatch,reason}) {
        for(;*p;++p) {h^=uint8_t(*p);h*=1099511628211ull;}h^=255;h*=1099511628211ull;
    } return h;
}
}

extern "C" int cf_initialize(const char * model_path) {
    auto begin=Clock::now();
    try {
        require(!s.initialized,"only one native session engine per process is supported");s.initialized=true;
        std::string mode=env("CF_MODE","");require(mode=="cf"||mode=="reference","unknown execution mode");
        s.enabled=mode=="cf";s.kernel=env("CF_KERNEL","dotprod");s.report=env("CF_REPORT","");
        s.scheduler=env("CF_SCHEDULER","contiguous");
        require(s.scheduler=="contiguous"||s.scheduler=="round-robin","unknown decode scheduler");
        s.validate=std::string(env("CF_VALIDATE","0"))=="1";s.memory=env("CF_MEMORY","resident");
        s.capture_dir=env("CF_CAPTURE_DIR","");
        std::string schedule=env("CF_SCHEDULE","");
        if(!schedule.empty()) {
            require(s.enabled&&!s.validate,"schedule requires CF mode without duplicate validation");
            require(s.kernel!="lut","schedule supports tile4 layout only");
            std::istringstream parts(schedule);std::string part;
            while(std::getline(parts,part,',')) {
                require(part=="reference"||part=="dotprod"||part=="asm"||part=="asm-unroll2"||part=="asm-prefetch512","unsupported schedule entry");
                if(part.rfind("asm",0)==0)require(cf::has_asm(),"assembly schedule unsupported on CPU/build");
                s.schedule.push_back(part);
            }
            require(s.schedule.size()<=256,"schedule too long");
        }
        if(!s.capture_dir.empty())std::filesystem::create_directories(s.capture_dir);
        if(std::strlen(env("CF_LOGITS",""))) {s.logits=std::fopen(env("CF_LOGITS",""),"wb");require(s.logits,"cannot open logits file");}
        if(std::strlen(env("CF_ROUTES",""))) {s.routes=std::fopen(env("CF_ROUTES",""),"wb");require(s.routes,"cannot open routes file");}
        require(s.memory=="resident","constrained policy requires an accounted cgroup; resident only in v1 integration");
        require(s.kernel=="portable"||s.kernel=="dotprod"||s.kernel=="lut"||s.kernel=="asm"||s.kernel=="asm-unroll2"||s.kernel=="asm-prefetch512","unknown kernel");
        if(s.kernel.rfind("asm",0)==0) {
            s.asm_kernel=cf::resolve_asm(s.kernel=="asm-unroll2"?cf::AsmVariant::Unroll2:s.kernel=="asm-prefetch512"?cf::AsmVariant::Prefetch512:cf::AsmVariant::Baseline);
            require(s.asm_kernel!=nullptr,"requested assembly is not compiled or runtime dotprod unavailable; choose an explicit supported backend");
        }
        if(s.enabled) {
            auto path=std::filesystem::absolute(env("CF_PLAN",""));
            require(!std::string(env("CF_PLAN","")).empty(),"CF mode requires --cf-plan");
            require(std::filesystem::file_size(path)<16*1024*1024,"manifest too large");
            std::ifstream f(path);json j;f>>j;
            require(j.at("format")=="cf-engine-plan"&&j.at("version").is_number_integer()&&j.at("version")==1,"unsupported artifact version");
            require(j.at("inventory").is_array()&&j.at("inventory").size()<=10000,"invalid tensor inventory");
            std::map<std::string,const json *,std::less<>> inventory;
            for(const auto & t:j.at("inventory")) {
                std::string name=t.at("name");require(!inventory.count(name),"duplicate inventory tensor");inventory[name]=&t;
            }
            verify_source(j,model_path);
            for(const auto & t:j.at("tensors")) {
                Entry e;e.name=t.at("name");e.layout=t.at("layout");e.source_hash=t.at("source_sha256");
                require(e.source_hash.size()==64&&e.source_hash.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid source tensor digest");
                auto original=inventory.find(e.name);require(original!=inventory.end(),"selected tensor absent from inventory");
                for(const auto key:{"type","shape","bytes","source_sha256","source_file","source_offset"})
                    require(t.at(key)==original->second->at(key),std::string("selected/inventory mismatch: ")+key);
                auto dims=t.at("shape");require(dims.is_array()&&dims.size()==3,"invalid shape");
                for(auto d:dims)require(d.is_number_unsigned()||(d.is_number_integer()&&d.get<int64_t>()>0),"negative or invalid shape");
                e.k=dims[0];e.m=dims[1];e.e=dims[2];e.bytes=t.at("bytes");
                require(e.k&&e.k<=2816&&e.k%32==0&&e.m&&e.m<=2816&&e.m%4==0&&e.e&&e.e<=128,"unsupported shape");
                require(t.at("type")=="Q4_0","unsupported packed type");
                require(e.bytes==checked_mul(checked_mul(e.e,e.m/4),checked_mul(e.k/32,72)),"payload bounds mismatch");
                require(e.layout==(s.kernel=="lut"?"q4-bitplane4-v1":"q4-tile4-v1"),"layout/kernel mismatch");
                std::filesystem::path rel=t.at("file").get<std::string>();
                require(!rel.empty()&&!rel.is_absolute(),"invalid artifact path");
                auto base=std::filesystem::canonical(path.parent_path()),p=std::filesystem::canonical(base/rel);
                auto r=p.lexically_relative(base);require(!r.empty()&&*r.begin()!="..","artifact path escapes plan");
                require(std::filesystem::file_size(p)==e.bytes,"payload size mismatch");
                int fd=open(p.c_str(),O_RDONLY);require(fd>=0,"cannot open payload");
                e.data=mmap(nullptr,e.bytes,PROT_READ,MAP_SHARED,fd,0);close(fd);
                require(e.data!=MAP_FAILED,"payload mmap failed");
                if(digest(e.data,e.bytes)!=t.at("sha256").get<std::string>()) {munmap(e.data,e.bytes);throw std::runtime_error("packed SHA256 mismatch");}
                require(!s.entries.count(e.name),"duplicate tensor");s.mapped_packed+=e.bytes;s.entries.emplace(e.name,std::move(e));
            }
            require(!s.entries.empty(),"empty CF plan");
            s.q8.resize(s.max_vectors*s.max_blocks);
            if(s.kernel=="lut")s.lut.resize(s.max_vectors*s.max_blocks*128);
            if(s.validate)s.validation.resize(s.max_output);
        }
        s.stats.reserve(4096);s.init_ms=ms(begin,Clock::now());s.ready=true;
        std::fprintf(stderr,"CF_ENGINE mode=%s kernel=%s dotprod_available=%d packed_bytes=%zu init_ms=%.3f validate=%d\n",mode.c_str(),s.kernel.c_str(),cf::has_dotprod(),s.mapped_packed,s.init_ms,s.validate);
        return 0;
    } catch(const std::exception & e) {std::fprintf(stderr,"CF_INIT_ERROR %s\n",e.what());return 3;}
}

extern "C" bool cf_compute_mul_mat_id(const ggml_compute_params * p,ggml_tensor * dst) {
    Entry * e=eligible(dst);
    if(!e) {if(p->ith==0){s.dispatch="reference";s.reason=!s.request_cf?"scheduled_reference":s.enabled?"tensor_or_shape_not_in_plan":"reference_mode";}return false;}
    const auto b=dst->src[1],ids=dst->src[2];
    const size_t blocks=e->k/32,vectors=size_t(b->ne[1]*b->ne[2]);
    if(p->ith==0) {
        s.dispatch=s.validate?"cf+reference_validation":s.dispatch_name.c_str();s.reason="supported";
        if(ids->ne[1]==1) {e->decode_output_mod64=int(reinterpret_cast<uintptr_t>(dst->data)%64);e->decode_stride_mod64=int(dst->nb[1]%64);}
        if(!e->source_checked) {
            auto check_start=Clock::now();
            if(digest(dst->src[0]->data,e->bytes)!=e->source_hash)GGML_ABORT("CF source tensor identity mismatch: %s",e->name.c_str());
            e->source_checked=true;
            s.source_tensor_check_ms+=ms(check_start,Clock::now());
        }
        for(int64_t t=0;t<ids->ne[1];++t)for(int64_t i=0;i<ids->ne[0];++i) {
            int32_t id;memcpy(&id,static_cast<char *>(ids->data)+t*ids->nb[1]+i*ids->nb[0],4);
            if(id<0||size_t(id)>=e->e)GGML_ABORT("CF expert ID out of bounds");
        }
        if(!s.capture_dir.empty()&&!e->captured&&ids->ne[1]==1) {
            std::string base=s.capture_dir+"/"+e->name;
            std::ofstream input(base+".f32",std::ios::binary);
            for(int64_t i=0;i<b->ne[1];++i)input.write(static_cast<char *>(b->data)+i*b->nb[1],std::streamsize(e->k*4));
            json capture={{"weight_name",e->name},{"K",e->k},{"M",e->m},{"E",e->e},{"input_vectors",b->ne[1]},{"phase",s.phase},{"ids",json::array()}};
            for(int64_t i=0;i<ids->ne[0];++i) {int32_t id;memcpy(&id,static_cast<char *>(ids->data)+i*ids->nb[0],4);capture["ids"].push_back(id);}
            std::ofstream metadata(base+".json");metadata<<capture.dump(2)<<'\n';
            if(!input||!metadata)GGML_ABORT("CF capture write failed");
            e->captured=true;
        }
        s.prep_end=Clock::now();
    }
    ggml_barrier(p->threadpool);
    auto quant=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    for(size_t v=p->ith;v<vectors;v+=p->nth) {
        size_t t=v/size_t(b->ne[1]),i=v%size_t(b->ne[1]);
        const auto x=reinterpret_cast<float *>(static_cast<char *>(b->data)+t*b->nb[2]+i*b->nb[1]);
        quant(x,s.q8.data()+v*blocks,e->k);
        if(s.kernel=="lut")cf::build_lut(s.q8.data()+v*blocks,blocks,s.lut.data()+v*blocks*128);
    }
    ggml_barrier(p->threadpool);
    if(p->ith==0) {auto now=Clock::now();s.prep_ms=ms(s.prep_end,now);s.prep_end=now;
        s.scratch_peak=std::max(s.scratch_peak,vectors*blocks*(sizeof(cf::Q8Block)+(s.kernel=="lut"?256:0)));}
    const size_t tiles=e->m/4;
    // Prefill groups by expert and keeps a weight tile live over assigned tokens.
    // Decode does not scan unselected experts and has no future-token batch.
    if(ids->ne[1]==1) {
        size_t jobs=size_t(ids->ne[0])*tiles;
        const int workers=jobs<16?1:p->nth;
        const bool contiguous=s.scheduler=="contiguous";
        size_t first=contiguous?jobs*size_t(p->ith)/workers:size_t(p->ith);
        size_t end=contiguous?jobs*size_t(p->ith+1)/workers:jobs;
        size_t step=contiguous?1:size_t(workers);
        if(p->ith<workers)for(size_t job=first;job<end;job+=step) {
            size_t slot=job/tiles,tile=job%tiles;int32_t id;
            memcpy(&id,static_cast<char *>(ids->data)+slot*ids->nb[0],4);
            auto w=static_cast<uint8_t *>(e->data)+(size_t(id)*tiles+tile)*blocks*72;
            size_t v=slot%size_t(b->ne[1]);float out[4];
            if(s.kernel=="lut")cf::dot4_lut(w,s.q8.data()+v*blocks,s.lut.data()+v*blocks*128,blocks,out);
            else if(s.kernel=="portable")cf::dot4_portable(w,s.q8.data()+v*blocks,blocks,out);
            else if(s.asm_kernel)s.asm_kernel(w,s.q8.data()+v*blocks,blocks,out);
            else cf::dot4_arm(w,s.q8.data()+v*blocks,blocks,out);
            memcpy(static_cast<char *>(dst->data)+slot*dst->nb[1]+tile*16,out,16);
        }
    } else {
        for(size_t expert=0;expert<e->e;++expert) {
            // Bounded stack mapping, shared IDs are read-only; no hot-path allocation.
            uint16_t tokens[512],slots[512];size_t n=0;
            for(int64_t t=0;t<ids->ne[1];++t)for(int64_t i=0;i<ids->ne[0];++i) {
                int32_t id;memcpy(&id,static_cast<char *>(ids->data)+t*ids->nb[1]+i*ids->nb[0],4);
                if(size_t(id)==expert){tokens[n]=uint16_t(t);slots[n++]=uint16_t(i);}
            }
            if(!n)continue;
            for(size_t tile=p->ith;tile<tiles;tile+=p->nth) {
                auto w=static_cast<uint8_t *>(e->data)+(expert*tiles+tile)*blocks*72;
                for(size_t row=0;row<n;++row) {
                    size_t v=tokens[row]*size_t(b->ne[1])+slots[row]%size_t(b->ne[1]);float out[4];
                    if(s.kernel=="lut")cf::dot4_lut(w,s.q8.data()+v*blocks,s.lut.data()+v*blocks*128,blocks,out);
                    else if(s.kernel=="portable")cf::dot4_portable(w,s.q8.data()+v*blocks,blocks,out);
                    else if(s.asm_kernel)s.asm_kernel(w,s.q8.data()+v*blocks,blocks,out);
                    else cf::dot4_arm(w,s.q8.data()+v*blocks,blocks,out);
                    memcpy(static_cast<char *>(dst->data)+tokens[row]*dst->nb[2]+slots[row]*dst->nb[1]+tile*16,out,16);
                }
            }
        }
    }
    ggml_barrier(p->threadpool);
    if(p->ith==0)s.mul_ms=ms(s.prep_end,Clock::now());
    if(s.validate) {
        if(p->ith==0)for(int64_t t=0;t<ids->ne[1];++t)for(int64_t i=0;i<ids->ne[0];++i)
            memcpy(s.validation.data()+(t*ids->ne[0]+i)*e->m,static_cast<char *>(dst->data)+t*dst->nb[2]+i*dst->nb[1],e->m*4);
        ggml_barrier(p->threadpool);
        return false;
    }
    return true;
}

extern "C" void cf_validate_after(const ggml_compute_params * p,ggml_tensor * dst) {
    if(!s.validate||!eligible(dst))return;
    ggml_barrier(p->threadpool);
    if(p->ith==0) {
        double error=0,norm=0,maxdiff=0,maxref=0;size_t n=0;
        for(int64_t t=0;t<dst->ne[2];++t)for(int64_t i=0;i<dst->ne[1];++i) {
            auto ref=reinterpret_cast<float *>(static_cast<char *>(dst->data)+t*dst->nb[2]+i*dst->nb[1]);
            for(int64_t r=0;r<dst->ne[0];++r) {
                double a=ref[r],b=s.validation[n++],d=a-b;
                if(!std::isfinite(a)||!std::isfinite(b)){++s.validation_failures;continue;}
                error+=d*d;norm+=a*a;maxdiff=std::max(maxdiff,std::abs(d));maxref=std::max(maxref,std::abs(a));
            }
        }
        double rel=std::sqrt(error/std::max(norm,1e-30)),scaled=maxdiff/std::max(1.0,maxref);
        s.max_rel_l2=std::max(s.max_rel_l2,rel);s.max_scaled_abs=std::max(s.max_scaled_abs,scaled);
        s.validation_values+=n;if(rel>2e-5||scaled>2e-4)++s.validation_failures;
    }
    ggml_barrier(p->threadpool);
}

extern "C" void cf_profile_begin(const ggml_compute_params * p,ggml_tensor * t) {
    if(p->ith||!s.ready)return;
    if(s.routes&&t->op==GGML_OP_MUL_MAT_ID) {
        const auto ids=t->src[2];
        uint64_t header[]={std::strlen(t->src[0]->name),uint64_t(ids->ne[0]),uint64_t(ids->ne[1]),s.request_index};
        if(std::fwrite(header,sizeof(header),1,s.routes)!=1||std::fwrite(t->src[0]->name,1,header[0],s.routes)!=header[0])GGML_ABORT("CF route write failed");
        for(int64_t token=0;token<ids->ne[1];++token)for(int64_t slot=0;slot<ids->ne[0];++slot)
            if(std::fwrite(static_cast<char *>(ids->data)+token*ids->nb[1]+slot*ids->nb[0],4,1,s.routes)!=1)GGML_ABORT("CF route write failed");
    }
    if(t->op==GGML_OP_GET_ROWS&&t->src[0]&&std::strcmp(t->src[0]->name,"token_embd.weight")==0)
        s.phase=ggml_nelements(t->src[1])>1?"prefill":"decode";
    s.dispatch="reference";s.reason="not_owned";s.prep_ms=0;s.mul_ms=0;s.start=Clock::now();
}
extern "C" void cf_profile_end(const ggml_compute_params * p,ggml_tensor * t,int fused) {
    if(p->ith||!s.ready)return;
    auto end=Clock::now();auto key=stat_key(t,s.dispatch,s.reason);
    if(std::strcmp(t->name,"attn_out-0")==0) {s.ffn_start=end;s.ffn_active=true;}
    if(std::strcmp(t->name,"l_out-0")==0&&s.ffn_active) {
        (s.phase=="prefill"?s.request_ffn_prefill_ms:s.request_ffn_decode_ms)+=ms(s.ffn_start,end);s.ffn_active=false;
    }
    auto it=s.stats.find(key);
    if(it==s.stats.end()) {
        if(s.stats.size()>=8192)GGML_ABORT("CF stats capacity exceeded");
        Stat st;st.name=t->name;st.op=ggml_op_name(t->op);st.phase=s.phase;st.dispatch=s.dispatch;st.reason=s.reason;
        if((t->op==GGML_OP_MUL_MAT||t->op==GGML_OP_MUL_MAT_ID)&&t->src[0])st.weight_name=t->src[0]->name;
        it=s.stats.emplace(key,std::move(st)).first;
    }
    auto & st=it->second;++st.calls;st.elapsed_ms+=ms(s.start,end);st.prepare_ms+=s.prep_ms;st.multiply_ms+=s.mul_ms;
    if(std::strcmp(s.dispatch,"reference")) {s.request_cf_ms+=ms(s.start,end);++s.request_cf_calls;}
    else {s.request_ref_ms+=ms(s.start,end);++s.request_ref_calls;}
    s.request_prepare_ms+=s.prep_ms;s.request_multiply_ms+=s.mul_ms;
    if(std::strcmp(s.dispatch,"reference"))++s.cf_calls;else ++s.fallback_calls;
    if(t->op==GGML_OP_MUL_MAT_ID)st.logical_weight_bytes+=ggml_nbytes(t->src[0])/size_t(t->src[0]->ne[2])*size_t(t->src[2]->ne[0])*size_t(t->src[2]->ne[1]);
    (void)fused; // A fused group is timed under its leading op; report states this.
}

extern "C" int cf_finish() {
    try {
        json j={{"format","cf-engine-run"},{"version",1},{"reference_revision","a7b94df2c616bc1f62a73b964b4a71cb0dcc488e"},
            {"execution_contract",s.validate?"operator_validation_reference_outputs":"semantics_preserving_candidate"},
            {"mode",s.enabled?"cf":"reference"},{"kernel_requested",s.kernel},{"kernel_executed",!s.cf_calls?std::string("none"):s.asm_kernel?"cf-"+s.kernel:s.kernel=="dotprod"?(cf::has_dotprod()?"cf-arm-dotprod":"cf-portable"):s.kernel=="lut"?"cf-bitplane-lut":"cf-portable"},
            {"dotprod_available",cf::has_dotprod()},{"memory_policy",s.memory},{"init_and_identity_check_ms",s.init_ms},
            {"mapped_packed_bytes",s.mapped_packed},{"source_file_bytes",s.source_bytes},
            {"scratch_allocated_bytes",s.q8.size()*sizeof(cf::Q8Block)+s.lut.size()*2+s.validation.size()*4},
            {"scratch_active_peak_bytes",s.scratch_peak},{"cf_calls",s.cf_calls},{"reference_calls",s.fallback_calls},
            {"first_use_source_tensor_identity_ms",s.source_tensor_check_ms},
            {"validation",{{"enabled",s.validate},{"values",s.validation_values},{"failures",s.validation_failures},{"max_relative_l2",s.max_rel_l2},{"max_scaled_absolute",s.max_scaled_abs},{"relative_l2_tolerance",2e-5},{"scaled_absolute_tolerance",2e-4}}},
            {"timing_note","Wall time per leading graph op includes existing worker barrier; fused groups attributed to leading op. Instrumentation overhead present in both paths."},
            {"phase_note","Profile plan label uses token-embedding row count: >1 prefill, otherwise decode. One-token reused/new input also uses the decode plan; request timings report input processing separately."},
            {"traffic_note","logical_weight_bytes is a modeled access count, NOT measured DRAM/cache/storage traffic"}};
        j["ops"]=json::array();double cf_ms=0,ref_ms=0;
        j["decode_scheduler"]=s.scheduler;
        j["decode_output_alignment"]=json::array();
        for(const auto & item:s.entries)j["decode_output_alignment"].push_back({{"tensor",item.first},{"address_mod64",item.second.decode_output_mod64},{"expert_stride_mod64",item.second.decode_stride_mod64}});
        j["request_execution"]=s.requests;
        if(!s.schedule.empty()) {j["kernel_executed"]="explicit-request-schedule";j["schedule"]=s.schedule;}
        for(const auto & item:s.stats) {const auto & st=item.second;
            j["ops"].push_back({{"name",st.name},{"weight_name",st.weight_name},{"op",st.op},{"phase",st.phase},{"dispatch",st.dispatch},{"reason",st.reason},{"calls",st.calls},{"elapsed_ms",st.elapsed_ms},{"prepare_ms",st.prepare_ms},{"multiply_ms",st.multiply_ms},{"logical_weight_bytes",st.logical_weight_bytes}});
            (st.dispatch=="reference"?ref_ms:cf_ms)+=st.elapsed_ms;
        }
        j["cf_op_ms"]=cf_ms;j["reference_op_ms"]=ref_ms;j["reference_runtime_share"]=cf_ms+ref_ms?ref_ms/(cf_ms+ref_ms):1;
        j["reference_runtime_share_excluding_first_use_hash"]=cf_ms+ref_ms>s.source_tensor_check_ms?ref_ms/(cf_ms+ref_ms-s.source_tensor_check_ms):1;
        if(s.validate) {j["reference_runtime_share"]=nullptr;j["validation_timing_note"]="CF op timing includes duplicate reference execution; runtime share is unavailable";}
        if(s.validate)j["reference_runtime_share_excluding_first_use_hash"]=nullptr;
        rusage usage{};getrusage(RUSAGE_SELF,&usage);j["peak_rss_kib"]=usage.ru_maxrss;j["major_faults"]=usage.ru_majflt;
        for(const char * name:{"status","io","smaps_rollup"}) {std::ifstream f(std::string("/proc/self/")+name);j["proc"][name]=std::string(std::istreambuf_iterator<char>(f),{});}
        if(s.logits) {require(std::fclose(s.logits)==0,"logits close failed");s.logits=nullptr;}
        if(s.routes) {require(std::fclose(s.routes)==0,"routes close failed");s.routes=nullptr;}
        if(!s.report.empty()) {std::ofstream f(s.report);require(bool(f),"cannot open report");f<<j.dump(2)<<'\n';require(bool(f),"report write failed");}
        std::fprintf(stderr,"CF_COVERAGE cf_calls=%llu reference_calls=%llu reference_runtime_share=%s validation_failures=%llu\n",(unsigned long long)s.cf_calls,(unsigned long long)s.fallback_calls,j["reference_runtime_share"].dump().c_str(),(unsigned long long)s.validation_failures);
        for(auto & it:s.entries)if(it.second.data!=MAP_FAILED)munmap(it.second.data,it.second.bytes);
        if(s.enabled&&!s.cf_calls) {std::fprintf(stderr,"CF_ERROR zero CF dispatches; run is not CF evidence\n");return 5;}
        return s.validation_failures?6:0;
    } catch(const std::exception & e) {std::fprintf(stderr,"CF_REPORT_ERROR %s\n",e.what());return 3;}
}

extern "C" void cf_record_logits(const float * logits,int vocab,unsigned long long request,unsigned long long token) {
    if(!s.logits||!logits)return;
    uint64_t header[]={uint64_t(vocab),uint64_t(request),uint64_t(token)};
    if(std::fwrite(header,sizeof(header),1,s.logits)!=1||std::fwrite(logits,sizeof(float),size_t(vocab),s.logits)!=size_t(vocab))
        GGML_ABORT("CF logits write failed");
}

extern "C" void cf_request_begin(unsigned long long request) {
    s.request_index=request;s.request_read_bytes=storage_read_bytes();
    rusage usage{};getrusage(RUSAGE_SELF,&usage);s.request_major_faults=usage.ru_majflt;
    if(!s.schedule.empty()) {
        if(request>=s.schedule.size())GGML_ABORT("CF schedule shorter than request list");
        const auto & mode=s.schedule[size_t(request)];s.request_cf=mode!="reference";
        s.kernel=mode;s.asm_kernel=nullptr;
        if(mode.rfind("asm",0)==0)s.asm_kernel=cf::resolve_asm(mode=="asm-unroll2"?cf::AsmVariant::Unroll2:mode=="asm-prefetch512"?cf::AsmVariant::Prefetch512:cf::AsmVariant::Baseline);
        s.dispatch_name="cf-"+mode;
    }
    s.request_cf_ms=s.request_ref_ms=s.request_prepare_ms=s.request_multiply_ms=0;
    s.request_cf_calls=s.request_ref_calls=0;s.request_hash_start=s.source_tensor_check_ms;
    s.request_ffn_prefill_ms=s.request_ffn_decode_ms=0;s.ffn_active=false;
}

extern "C" void cf_request_end(unsigned long long request) {
    rusage usage{};getrusage(RUSAGE_SELF,&usage);
    s.requests.push_back({{"request",request},{"kernel",s.enabled&&s.request_cf?s.kernel:"reference"},
        {"cf_calls",s.request_cf_calls},{"reference_calls",s.request_ref_calls},{"cf_ms",s.request_cf_ms},
        {"reference_ms",s.request_ref_ms},{"prepare_ms",s.request_prepare_ms},{"multiply_ms",s.request_multiply_ms},
        {"source_check_ms",s.source_tensor_check_ms-s.request_hash_start},
        {"layer0_complete_ffn_prefill_ms",s.request_ffn_prefill_ms},{"layer0_complete_ffn_decode_ms",s.request_ffn_decode_ms},
        {"storage_read_bytes",storage_read_bytes()-s.request_read_bytes},{"major_faults",usage.ru_majflt-s.request_major_faults},
        {"block_boundary","after attn_out-0 through l_out-0; shared FFN, expert FFN, norms, routing, combination, residual and layer scale"}});
}
