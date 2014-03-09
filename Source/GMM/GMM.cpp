/*
 *
 * EM: 
 * pick starting a, u and C for each G_k
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

inline double GMM::ComputeWeightedGaussian(const cv::Mat &x, WeightedGaussian wg)
{
    return GMM::ComputeWeightedGaussian(x, 
                                        std::get<0>(wg),
                                        std::get<1>(wg),
                                        std::get<2>(wg));
}

inline double GMM::ComputeWeightedGaussian(const cv::Mat &x, double weight, const cv::Mat &mean, const cv::Mat &covariance)
{
    cv::Mat precision;
    cv::invert(covariance, precision);
    
    double detp = cv::determinant(precision);
    
    double outter = std::sqrt(detp / 2.0 * M_PI);
    
    cv::Mat meanDist = x - mean;
    cv::Mat meanDistTrans;
    cv::transpose(meanDist, meanDistTrans);
    
    cv::Mat symmetricProduct = meanDistTrans * precision * meanDist; // This is a "1x1" matrix e.g. a scalar value
    double inner = symmetricProduct.at<double>(0,0) / -2.0;
    
    return weight * outter * std::exp(inner);
}

inline double GMM::operator ()(const cv::Mat &x) const
{
    return std::accumulate(_Gaussians.begin(), _Gaussians.end(), 0, [&x](double val, WeightedGaussian wg) { return val + ComputeWeightedGaussian(x, wg); });
}
