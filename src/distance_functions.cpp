#include <vector>
#include "distance_functions.h"
#include <immintrin.h>

float distance_functions_t::compute(distance_metric_t metric, const std::vector<float>&a, const std::vector<float>&b, size_t dims) {
    switch(metric) {
        case distance_metric_t::L2:
            return l2_distance(a, b, dims);
        case distance_metric_t::INNER_PRODUCT:
            return ip_distance(a, b, dims);
        default:
            return -1.0f;
    }
}

float distance_functions_t::l2_distance_plain(const std::vector<float>&a, const std::vector<float>&b, size_t dim) {
    float dist = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += (diff * diff);
    }
    return dist;
}


float distance_functions_t::l2_distance(const std::vector<float>& a, const std::vector<float>& b, size_t dim) {
    const float* x = a.data();
    const float* y = b.data();
    float dist = 0.0f;

    // Process 8 floats at a time using AVX
    size_t i = 0;
    const size_t limit = dim - (dim % 8);

    if (limit > 0) {
        __m256 sum = _mm256_setzero_ps();

        for (; i < limit; i += 8) {
            // Load 8 floats from each vector
            __m256 va = _mm256_loadu_ps(x + i);
            __m256 vb = _mm256_loadu_ps(y + i);

            // Compute difference
            __m256 diff = _mm256_sub_ps(va, vb);

            // Square and accumulate
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        // Horizontal sum using simple extraction
        float temp[8];
        _mm256_storeu_ps(temp, sum);
        dist = temp[0] + temp[1] + temp[2] + temp[3] +
               temp[4] + temp[5] + temp[6] + temp[7];
    }

    // Handle remaining elements
    for (; i < dim; i++) {
        float diff = x[i] - y[i];
        dist += diff * diff;
    }

    return dist;
}

float distance_functions_t::ip_distance(const std::vector<float>&a, const std::vector<float>&b, size_t dim) {
    float dist = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dist += a[i] * b[i];
    }
    return 1.0f - dist;
}
