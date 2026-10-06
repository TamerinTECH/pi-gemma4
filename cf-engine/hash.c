// Public-domain SHA256 reused from the pinned llama.cpp vendor tree.
#define sha256_init cf_sha256_init
#define sha256_update cf_sha256_update
#define sha256_final cf_sha256_final
#define sha256_hash cf_sha256_hash
#include "hash/sha256/sha256.c"
