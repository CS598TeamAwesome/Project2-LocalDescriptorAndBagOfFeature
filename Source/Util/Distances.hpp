#pragma once
#include <opencv2/opencv.hpp>
#include <exception>
#include <vector>
#include <array>
#include <functional>
#include <assert.h>
#include <stdlib.h>
#include <numeric>

namespace LocalDescriptorAndBagOfFeature {

    double euclidean_distance(const std::vector<double> &v1, const std::vector<double> &v2);
    void vector_add(std::vector<double> &v1, std::vector<double> &v2);
    void vector_subtract(std::vector<double> &v1, std::vector<double> &v2);
    void convert_mat_to_vector(const cv::Mat &descriptors, std::vector<std::vector<double>> &samples);
}
