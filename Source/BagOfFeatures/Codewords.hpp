#pragma once
#include <vector>

namespace LocalDescriptorAndBagOfFeature 
{
    void FindCodewords(const std::vector<std::vector<double>> &features, int numCodeWords, std::vector<std::vector<double>> &codewords);
}
