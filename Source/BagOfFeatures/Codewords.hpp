#pragma once
#include <vector>
#include <string>

namespace LocalDescriptorAndBagOfFeature 
{
    void FindCodewords(const std::vector<std::vector<double>> &features, int numCodeWords, std::vector<std::vector<double>> &codewords);
    void FindCodewords(const std::vector<std::vector<double>> &features, int numCodeWords, std::vector<std::vector<double>> &codewords, int iterationCap, int epsilon, int trials);
    void SaveCodebook(std::string filename, const std::vector<std::vector<double>> &codebook);
    void LoadCodebook(std::string filename, std::vector<std::vector<double>> &codebook);
}
