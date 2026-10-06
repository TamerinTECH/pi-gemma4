#pragma once
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
struct ggml_compute_params;
struct ggml_tensor;
bool cf_compute_mul_mat_id(const struct ggml_compute_params *, struct ggml_tensor *);
void cf_validate_after(const struct ggml_compute_params *, struct ggml_tensor *);
void cf_profile_begin(const struct ggml_compute_params *, struct ggml_tensor *);
void cf_profile_end(const struct ggml_compute_params *, struct ggml_tensor *, int n_fused);
int cf_initialize(const char * model_path);
int cf_finish(void);
void cf_record_logits(const float *, int vocab, unsigned long long request, unsigned long long token);
void cf_request_begin(unsigned long long request);
void cf_request_end(unsigned long long request);
#ifdef __cplusplus
}
#endif
