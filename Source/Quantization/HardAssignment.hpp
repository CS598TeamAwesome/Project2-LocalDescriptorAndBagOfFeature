#pragma once
#include <opencv2/opencv.hpp>
#include <exception>
#include <vector>
#include <cmath>
#include <array>
#include <functional>
#include <assert.h>
#include <stdlib.h>
#include "Quantization.hpp"
#include "../Util/Distances.hpp"

namespace LocalDescriptorAndBagOfFeature {

    class HardAssignment : public Quantization {
        public:
            HardAssignment(std::vector<std::vector<double>> codebook);
            int nearest_codeword(const std::vector<double> &region);
            void quantize(const std::vector<std::vector<double>> &regions, std::vector<double> &histogram);

        private:
            std::vector<std::vector<double>> codebook;
    };
}
