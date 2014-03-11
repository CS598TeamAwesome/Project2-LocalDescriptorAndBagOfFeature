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
 * (these equations are probably all wrong)
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

// Evaluates the weighted multivariate gaussian for a sample point
double WeightedGaussian::operator ()(const cv::Mat &x) const
{    
    cv::Mat precision = _Covariance.inv();    
    double outer = std::sqrt(cv::determinant(precision) / 2.0 * M_PI);
    
    cv::Mat meanDist = x - _Mean;    
    cv::Mat symmetricProduct = meanDist.t() * precision * meanDist; // This is a "1x1" matrix e.g. a scalar value
    double inner = symmetricProduct.at<double>(0,0) / -2.0;
    
    return _Weight * outer * std::exp(inner);
}

GMM::GMM(int num, double convergenceThreshold)
    : _Gaussians(num), _ConvergenceThreshold(convergenceThreshold)
{
    
}

// Trains a GMM on feature data from a set of images using the Expecation Maximization (EM) algorithm
// This is probably pretty slow
void GMM::Train(const FeatureSet &featureSet, int maxIterations)
{
    // Make one long vector out of the input features
    BagOfFeatures allFeatures;
    for(BagOfFeatures singleImg : featureSet)
    {
        for(Histogram h : singleImg)
        {
            allFeatures.push_back(h);
        }
    }
    
    // Make some initialization using k-means
    double llikelihood = _LogLikelihood(allFeatures);
    
    for(int i = 0; i < maxIterations; i++)
    {       
        cv::Mat gamma = _E(allFeatures);
        _M(gamma, allFeatures);
        
        if((_LogLikelihood() - llikelihood) <= _ConvergenceThreshold)
            break;
    }
}

// Finds the responsibility of the kth gaussian for a sample
// Posterior probability that x belongs to the kth gaussian found using bayes theorem
double GMM::_Responsibility(const cv::Mat &x, int k) const
{
    const GMM &model = *this;
    return _Gaussians[k](x) / model(x);
}

// E - step
// Computes the responsibility of each gaussian for each data sample
// returns an nxk matrix (the gamma matrix) where n is the number of features and k is the number of gaussians
cv::Mat GMM::_E(const BagOfFeatures &bof) const
{
    cv::Mat gamma(bof.size(), _Gaussians.size(), CV_64F);
    
    for(int n = 0; n < bof.size(); n++) // For each sample
    {
        for(int k = 0; k < _Gaussians.size(); k++) // and each gaussian
        {
            gamma.at<double>(n, k) = _Responsibility(cv::Mat(bof[n]), k); // find its responsibility
        }
    }
    
    return gamma;
}

// M - step
// Improves the mean, covariance, and weight of each gaussian using
// the responsibilites computed in the E step
void GMM::_M(const cv::Mat &gamma, const BagOfFeatures &bof)
{
    for(int k = 0; k < _Gaussians.size(); k++)
    {
        // Get the kth gaussian responsibilities
        cv::Mat gammak = gamma.col(k);
        
        // Compute Nk, the sum of all responsibilties for this gaussian
        double Nk = std::accumulate(gammak.begin<double>(), gammak.end<double>(), 0);
        
        // Update the mean
        cv::Mat uNew = cv::Mat::zeros(gammak.size(), CV_64F);
        for(int n = 0; n < gammak.rows; n++)
        {
            uNew += gammak.at<double>(n, 1) * cv::Mat(bof[n]);
        }
        
        uNew /= Nk;
        _Gaussians[k].Mean() = uNew;
        
        // Update the covariance
        cv::Mat sigmaNew = cv::Mat::zeros(gammak.rows, gammak.rows, CV_64F);
        for(int n = 0; n < gammak.rows; n++)
        {
            cv::Mat meanDistance = cv::Mat(bof[n]) - uNew;
            sigmaNew += gammak.at<double>(n, 1) * (meanDistance * meanDistance.t());
        }
        
        sigmaNew /= Nk;
        _Gaussians[k].Covariance() = sigmaNew;
        
        // Udpate weight
        _Gaussians[k].Weight() = Nk / bof.size();
    }
}

// Computes the likelihood that the GMM that is currently trained
// is the one that generated our data
double GMM::_LogLikelihood(const BagOfFeatures &bof) const
{
    const GMM &gmm = *this;
    return std::accumulate(bof.begin(), bof.end(), 0, [&gmm](double val, const Histogram &sample) { return val + std::log(gmm(cv::Mat(sample))); });
}

// Evaluates the GMM on a sample point
double GMM::operator ()(const cv::Mat &x) const
{
    return std::accumulate(_Gaussians.begin(), _Gaussians.end(), 0, [&x](double val, WeightedGaussian wg) { return val + wg(x); });
}
