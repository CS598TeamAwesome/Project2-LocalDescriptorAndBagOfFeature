#include "Codewords.hpp"
#include "../Util/Clustering.hpp"

void LocalDescriptorAndBagOfFeature::FindCodewords(const std::vector<std::vector<double> > &features, int numCodeWords, std::vector<std::vector<double> > &codewords)
{
    std::vector<int> labels, sizes;
    kmeans(features, numCodeWords, labels, codewords, sizes, 30);
}
