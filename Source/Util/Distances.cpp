#include "Distances.hpp"

void LocalDescriptorAndBagOfFeature::vector_add(std::vector<double> &v1, std::vector<double> &v2){
    assert(v1.size() == v2.size());

    std::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), std::plus<double>());
}

void LocalDescriptorAndBagOfFeature::vector_subtract(std::vector<double> &v1, std::vector<double> &v2){
    assert(v1.size() == v2.size());

    std::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), std::minus<double>());
}

double LocalDescriptorAndBagOfFeature::euclidean_distance(const std::vector<double> &v1, const std::vector<double> &v2){
    assert(v1.size() == v2.size());

    //get differences
    std::vector<double> diff(v1.size());
    std::transform(v1.begin(), v1.end(), v2.begin(), diff.begin(), std::minus<double>());

    //sum squares and take sqrt
    double sum2 = std::accumulate(diff.begin(), diff.end(), 0.0, [](double accum, double elem) { return accum + elem * elem; });
    double sum = std::sqrt(sum2);

    return sum;
}
