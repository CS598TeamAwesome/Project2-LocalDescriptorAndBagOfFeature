#pragma once
#include <opencv2/opencv.hpp>
#include <exception>
#include <vector>
#include <cmath>
#include <array>
#include <functional>
#include <assert.h>
#include <stdlib.h>
#include "../Util/Distances.hpp"

#ifndef M_PI
#   define M_PI 3.14159265358979323846
#endif

namespace LocalDescriptorAndBagOfFeature {

    const double normal_coefficient = 1/(std::sqrt(2*M_PI));
    double gaussian_kernel(double sigma, double x);

    class CodewordUncertainty { //potentially want a Quantization superclass
        public:
            CodewordUncertainty(std::vector<std::vector<double>> codebook, double sigma);
            void quantize(const std::vector<double> &region, std::vector<double> &histogram);
            void quantize(const std::vector<std::vector<double>> &regions, std::vector<double> &histogram);

        private:
            std::vector<std::vector<double>> codebook;
            double sigma;
    };
}
