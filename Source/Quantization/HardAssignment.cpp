#include "HardAssignment.hpp"
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <limits>

using namespace LocalDescriptorAndBagOfFeature;

//codebook is the vocabulary
HardAssignment::HardAssignment(std::vector<std::vector<double>> codebook):codebook(codebook){
}

//take a region and return the index of the nearest codeword
int HardAssignment::nearest_codeword(const std::vector<double> &region){

    int closest_index = 0;
    double closest_distance = euclidean_distance(codebook[0], region);

    for(int i = 1; i < codebook.size(); i++){
        double distance = euclidean_distance(codebook[i], region);
        if(distance < closest_distance){
            closest_index = i;
            closest_distance = distance;
        }
    }

    return closest_index;
}

//return the histogram of features for the regions in an image
void HardAssignment::quantize(const std::vector<std::vector<double>> &regions, std::vector<double> &histogram){
    std::vector<double> partial_histogram;
    histogram.clear();
    histogram.resize(codebook.size());

    //loop over regions, incrementing the corresponding histogram value for each region
    for(const std::vector<double>& region : regions){
        histogram[nearest_codeword(region)]++;
    }
}
