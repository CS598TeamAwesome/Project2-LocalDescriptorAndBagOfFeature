#include "CodewordUncertainty.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <limits>

using namespace LocalDescriptorAndBagOfFeature;

//gaussian-shaped kernel, we pair it with Euclidean distance (Gemert et al.)
double LocalDescriptorAndBagOfFeature::gaussian_kernel(double sigma, double x){
    return normal_coefficient * 1/sigma * std::exp((-1*x*x)/(2*sigma*sigma));
}

//sigma is a smoothing parameter and codebook is the vocabulary (Gemert et al.)
CodewordUncertainty::CodewordUncertainty(std::vector<std::vector<double>> codebook, double sigma):codebook(codebook){
    this->sigma = sigma;
}

//take a region and quantize it to the feature space using codeword uncertainty (Gemert et al.)
void CodewordUncertainty::quantize_region(const std::vector<double> &region, std::vector<double> &histogram){
    //compute distance table of codevectors to region
    std::vector<double> distances;
    for(std::vector<double>& codeword : codebook){
        double distance = euclidean_distance(codeword, region);
        distances.push_back(distance);
    }

    //compute normalization factor
    double norm = std::accumulate(distances.begin(), distances.end(), 0.0, [&](double accum, double elem) { return accum + gaussian_kernel(sigma, elem); });

    //compute histogram values
    histogram.clear();
    for(double& d : distances){
        double h_value = gaussian_kernel(sigma, d)/norm;
        histogram.push_back(h_value);
    }
}

//return the histogrammatic quantization for all the regions from an image
void CodewordUncertainty::quantize(const std::vector<std::vector<double>> &regions, std::vector<double> &histogram){
    std::vector<double> partial_histogram;
    histogram.clear();
    histogram.resize(codebook.size());

    //loop over regions, adding the partial histogram to the aggregate histogram
    for(const std::vector<double>& region : regions){
        quantize_region(region, partial_histogram);
        vector_add(histogram, partial_histogram);
    }
}
