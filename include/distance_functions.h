#pragma once

#include <cstddef>
#include <string>


enum class distance_metric_t {
    L2,
    INNER_PRODUCT
};

class distance_functions_t {
public:
    static float l2_distance_plain(const std::vector<float>&a, const std::vector<float>&b, size_t dim);

    static float l2_distance(const std::vector<float>&a, const std::vector<float>&b, size_t dim);

    static float ip_distance(const std::vector<float>&a, const std::vector<float>&b, size_t dim);

    static float compute(distance_metric_t metric, const std::vector<float>&a, const std::vector<float>&b, size_t dims);
};
