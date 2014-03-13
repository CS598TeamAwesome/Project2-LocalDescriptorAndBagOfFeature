#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include "../Util/Types.hpp"

namespace LocalDescriptorAndBagOfFeature 
{   
    class WeightedGaussian
    {
        public:
            WeightedGaussian(void);
            WeightedGaussian(double weight, const cv::Mat &mean, const cv::Mat &covariance);
            
            double operator ()(const cv::Mat &x) const;
            
            double &Weight(void) { return _Weight; }
            cv::Mat &Mean(void) { return _Mean; }
            cv::Mat &Covariance(void) { return _Covariance; }
            
        private:
            double _Weight;
            cv::Mat _Mean;
            cv::Mat _Covariance;
    };
    
    class GMM
    {
        public:
            GMM(int num);
            
            void Train(const FeatureSet &featureSet, int maxIterations);
            std::vector<double> Supervector(const BagOfFeatures &bof) const;
            
            int NumGaussians(void) const { return _Gaussians.size(); }
            
            double operator ()(const cv::Mat &x) const;
            
        private:                                    
            double _Responsibility(const cv::Mat &x, int k) const;
            cv::Mat _E(const BagOfFeatures &bof) const;
            void _M(const cv::Mat &gamma, const BagOfFeatures &bof);
            double _LogLikelihood(void) const;
            
            std::vector<WeightedGaussian> _Gaussians;
    };
}
