#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "../Quantization/Quantization.hpp"
#include "../Util/Types.hpp"

namespace LocalDescriptorAndBagOfFeature
{
    void train_category(const std::vector<cv::Mat> &samples, Histogram &centroid, const cv::Ptr<cv::FeatureDetector> &detector, const cv::SiftDescriptorExtractor &extractor, Quantization *quant);

    int get_category(const Histogram &feature_vector, const std::vector<std::vector<double>> &category_centroids);
    void test_category(std::vector<Histogram> &feature_vectors, std::vector<double> &confusion_table, std::vector<std::string> &category_labels, std::vector<std::vector<double>> &category_centroids);

    void save_classifier(std::string filename, const std::vector<std::vector<double>> &category_centroids, const std::vector<std::string> &category_labels);
    void load_classifier(std::string filename, std::vector<std::string> &category_labels, std::vector<std::vector<double>> &category_centroids);
}
