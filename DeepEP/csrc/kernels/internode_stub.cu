#include <cstdint>

#include "api.cuh"
#include "exception.cuh"

namespace deep_ep::internode {

namespace {

void unsupported() {
    EP_HOST_ASSERT(false and
                   "SM80 DeepEP build does not support multi-node normal mode");
}

}  // namespace

int get_source_meta_bytes() {
    // Keep the normal-mode buffer layout compatible with SourceMeta.
    return 2 * sizeof(int);
}

void notify_dispatch(
    const int*, int*, int, const int*, int*, const int*, int*, int,
    const bool*, int, int, int, int, int, int, int*, int*, int*, int*, void*,
    int, void**, int, int**, int, cudaStream_t, int64_t, int64_t, bool) {
    unsupported();
}

void dispatch(
    void*, float*, int64_t*, float*, void*, const void*, const float*,
    const int64_t*, const float*, int*, int*, int*, int*, const int*,
    const int*, const int*, const int*, const bool*, int, int, int, int, int,
    int, int, void*, int, int, void**, int, int, int, int, bool, cudaStream_t,
    int, bool) {
    unsupported();
}

void cached_notify(
    int, int, int, int, int, int, int, int*, const int*, const int*, int*,
    void*, int, void**, int, int**, int, cudaStream_t, int64_t, int64_t, bool,
    bool) {
    unsupported();
}

void combine(
    cudaDataType_t, void*, float*, const bool*, const void*, const float*,
    const void*, const void*, const int*, const int*, const void*, const int*,
    const int*, const int*, int, int, int, int, void*, int, int, void**, int,
    int, int, int, cudaStream_t, int, bool) {
    unsupported();
}

}  // namespace deep_ep::internode
