#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include "../Util/Types.hpp"

namespace LocalDescriptorAndBagOfFeature 
{   
    // Weighted gaussian is defined as a (weight, mean vector, covariance matrix)
    typedef std::tuple<double, cv::Mat, cv::Mat> WeightedGaussian; 
    
    class GMM
    {
        public:
            GMM(int numGaussians);
            
            void Train(const FeatureSet &featureSet);
            std::vector<double> Supervector(const BagOfFeatures &bof);
            
            int NumGaussians(void) const;
            
            double operator ()(const cv::Mat &x) const;
            
        private:                        
            static double ComputeWeightedGaussian(const cv::Mat &x, WeightedGaussian wg);
            static double ComputeWeightedGaussian(const cv::Mat &x, double weight, const cv::Mat &mean, const cv::Mat &covariance);
            
            std::vector<WeightedGaussian> _Gaussians;
            
            int _NumGaussians;
    };
}
