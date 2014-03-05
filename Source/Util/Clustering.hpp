#pragma once
#include <opencv2/opencv.hpp>
#include <exception>
#include <vector>
#include <array>
#include <functional>
#include <assert.h>

namespace LocalDescriptorAndBagOfFeature {
    //input is a vector of the samples (each a vector<double>) to cluster
    //k is the number of clusters
    std::vector<std::vector<double>> kmeans(std::vector<std::vector<double>> &input, int k);
}
