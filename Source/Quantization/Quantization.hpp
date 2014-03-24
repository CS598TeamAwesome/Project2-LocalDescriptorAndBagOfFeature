#pragma once
#include <vector>

namespace LocalDescriptorAndBagOfFeature
{
    class Quantization
    {
        public:
            virtual void quantize(const std::vector<std::vector<double>> &regions, std::vector<double> &histogram)=0;
    };
}
