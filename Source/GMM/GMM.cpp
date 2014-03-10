/*
 *
 * EM: 
 * pick starting a, u and C for each G_k (can use k-means for this somehow)
 *
 * update:
 * a_k = n_k / n
 * u_k = (1 / n_k) * sum (i, r_ik * x_i)
 * C_k = ( (1 / n_k) * sum(i, r_ik * x_i * x'_i) ) - (u_k * u'_k)
 *
 * Where: r_ik = (a_k * G(x_i | u_k, C_k)) / sum(l, a_l * G(x_i | u_l, C_l))
 *
 */
#include "GMM.hpp"
#include <cmath>

#ifndef M_PI
#   define M_PI 3.14159265358979323846
#endif

using namespace LocalDescriptorAndBagOfFeature;

WeightedGaussian::WeightedGaussian(double weight, const cv::Mat &mean, const cv::Mat &covariance)
    : _Weight(weight), _Mean(mean.clone()), _Covariance(covariance.clone())
{
    
}

double WeightedGaussian::operator ()(const cv::Mat &x) const
{    
    cv::Mat precision = _Covariance.inv();    
    double outer = std::sqrt(cv::determinant(precision) / 2.0 * M_PI);
    
    cv::Mat meanDist = x - _Mean;    
    cv::Mat symmetricProduct = meanDist.t() * precision * meanDist; // This is a "1x1" matrix e.g. a scalar value
    double inner = symmetricProduct.at<double>(0,0) / -2.0;
    
    return _Weight * outer * std::exp(inner);
}

GMM::GMM(int num)
    : _Gaussians(num)
{
    
}

void GMM::Train(const FeatureSet &featureSet)
{
    
}

double GMM::operator ()(const cv::Mat &x) const
{
    return std::accumulate(_Gaussians.begin(), _Gaussians.end(), 0, [&x](double val, WeightedGaussian wg) { return val + wg(x); });
}
