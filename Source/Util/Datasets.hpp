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
    void load_graz2_train(std::vector<std::vector<cv::Mat>> &images, std::vector<std::string> &labels);
    void load_graz2_validate(std::vector<std::vector<cv::Mat>> &images, std::vector<std::string> &labels);
    void load_graz2_test(std::vector<std::vector<cv::Mat>> &images, std::vector<std::string> &labels);

    void load_scene15_train(std::vector<std::vector<cv::Mat>> &images, std::vector<std::string> &labels);
    void load_scene15_test(std::vector<std::vector<cv::Mat>> &images, std::vector<std::string> &labels);
}
