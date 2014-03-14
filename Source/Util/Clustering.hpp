#pragma once
#include <opencv2/opencv.hpp>
#include <exception>
#include <vector>
#include <array>
#include <functional>
#include <assert.h>
#include <stdlib.h>
#include <numeric>
#include "Distances.hpp"

namespace LocalDescriptorAndBagOfFeature {

    //single run with randomized initial centers
    double kmeans(std::vector<std::vector<double>> input, int K, std::vector<int> &labels, std::vector<std::vector<double>> &centers, std::vector<int> &sizes);
    //multiple run, returning the best
    double kmeans(const std::vector<std::vector<double>> &input, int K, std::vector<int> &labels, std::vector<std::vector<double>> &centers, std::vector<int> &sizes, int trials);

}
