#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <cmath>
#include <time.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include "BagOfFeatures/Codewords.hpp"
#include "Classification/NearestCentroidClassifier.hpp"
#include "Quantization/CodewordUncertainty.hpp"
#include "Quantization/HardAssignment.hpp"
#include "Quantization/Quantization.hpp"
#include "Util/Datasets.hpp"
#include "Util/Distances.hpp"
#include "Util/Types.hpp"

using std::vector;
using namespace LocalDescriptorAndBagOfFeature;

void compute_histogram(cv::Mat &sample, Histogram &feature_vector, cv::Ptr<cv::FeatureDetector> &detector, cv::SiftDescriptorExtractor &extractor, Quantization *quant){
    //detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    detector->detect( sample, keypoints );

    //compute descriptor
    cv::Mat descriptor_uchar;
    extractor.compute(sample, keypoints, descriptor_uchar);

    cv::Mat descriptor_double;
    descriptor_uchar.convertTo(descriptor_double, CV_64F);

    //convert from mat to bag of unquantized features
    BagOfFeatures unquantized_features;
    convert_mat_to_vector(descriptor_double, unquantized_features);

    //quantize regions -- true BagOfFeatures
    quant->quantize(unquantized_features, feature_vector);
}

void compute_histograms(std::vector<cv::Mat> &samples, std::vector<Histogram> &feature_vectors, cv::Ptr<cv::FeatureDetector> &detector, cv::SiftDescriptorExtractor &extractor, Quantization *quant){
    int i = 0;
    for(cv::Mat& sample : samples){
        i++;
        //std::cout << "computing histogram for image " << i << " of " << samples.size() << std::endl;
        Histogram feature_vector;
        compute_histogram(sample, feature_vector, detector, extractor, quant);
        feature_vectors.push_back(feature_vector);
    }
}

int main(int argc, char **argv){

    //load test images -- TODO: may want to make some effort to generalize this to other datasets
    std::cout << "Load Test Images" << std::endl;
    std::vector<std::vector<cv::Mat>> test_images;
    std::vector<std::string> test_labels;
    load_graz2_validate(test_images, test_labels);

    //load codebook
    std::cout << "Load Codebook" << std::endl;
    std::vector<std::vector<double>> codebook;
    LoadCodebook("codebook_graz2_400_30.out", codebook);

    //load nearest centroid classifier
    std::cout << "Load Classifier" << std::endl;
    std::vector<std::string> category_labels;
    std::vector<std::vector<double>> category_centroids;
    load_classifier("graz2_centroid_classifier_400_30.out", category_labels, category_centroids);

    //TODO: vary these choices to compare performance
    cv::SiftFeatureDetector detector_sift(200); //sift-200 keypoints
    cv::SiftDescriptorExtractor extractor;      //sift128 descriptor

    //cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("Dense");
    //detector->set("featureScaleLevels", 1);
    //detector->set("featureScaleMul", 0.1f);
    //detector->set("initFeatureScale", 1.f);
    //detector->set("initXyStep", 30);

    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("SIFT");
    detector->set("nFeatures", 200);

    CodewordUncertainty soft_quant(codebook, 100.0);   //soft quantization
    HardAssignment hard_quant(codebook);

    Quantization *quant = &soft_quant;


    for(int i = 0; i < test_images.size(); i++){
        std::cout << "Compute Vectors for " << test_labels[i] << std::endl;
        std::vector<Histogram> feature_space_vectors;
        compute_histograms(test_images[i], feature_space_vectors, detector, extractor, quant);

        std::cout << "Categorize " << test_labels[i] << std::endl;
        std::vector<double> confusion_table(category_centroids.size());
        test_category(feature_space_vectors, confusion_table, category_labels, category_centroids);
    }

    return 0;
}
