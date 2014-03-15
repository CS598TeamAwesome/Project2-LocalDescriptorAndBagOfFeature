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
#include "Quantization/CodewordUncertainty.hpp"
#include "Quantization/HardAssignment.hpp"
#include "Util/Datasets.hpp"
#include "Util/Distances.hpp"
#include "Util/Types.hpp"

using std::vector;
using namespace LocalDescriptorAndBagOfFeature;

void convert_mat_to_vector(const cv::Mat &descriptors, std::vector<std::vector<double>> &samples){
    for(int i = 0; i < descriptors.rows; i++){
        //Mat to vector<double> conversion method as described in the OpenCV Documentation
        const double* p = descriptors.ptr<double>(i);
        std::vector<double> vec(p, p + descriptors.cols);
        samples.push_back(vec);
    }
}

//gets a centroid classifier trained from test data
void load_classifier(std::string filename, std::vector<std::string> &category_labels, std::vector<std::vector<double>> &category_centroids){
    //load codebook from file
    std::ifstream filein (filename);
    std::string s;
    std::getline(filein, s);
    std::istringstream sin(s);

    int category_ct;
    sin >> category_ct;

    for(int i = 0; i < category_ct; i++){
        getline(filein, s);
        category_labels.push_back(s);

        getline(filein, s);
        sin.str(s);
        sin.clear();

        vector<double> centroid;
        double d;
        while(sin >> d){
            centroid.push_back(d);
        }
        category_centroids.push_back(centroid);
    }
    filein.close();
}

void compute_histogram(cv::Mat &sample, Histogram &feature_vector, cv::SiftFeatureDetector &detector_sift, cv::SiftDescriptorExtractor &extractor, CodewordUncertainty hard_quant){
    //detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    detector_sift.detect( sample, keypoints );

    //compute descriptor
    cv::Mat descriptor_uchar;
    extractor.compute(sample, keypoints, descriptor_uchar);

    cv::Mat descriptor_double;
    descriptor_uchar.convertTo(descriptor_double, CV_64F);

    //convert from mat to bag of unquantized features
    BagOfFeatures unquantized_features;
    convert_mat_to_vector(descriptor_double, unquantized_features);

    //quantize regions -- true BagOfFeatures
    hard_quant.quantize(unquantized_features, feature_vector);
}

void compute_histograms(std::vector<cv::Mat> &samples, std::vector<Histogram> &feature_vectors, cv::SiftFeatureDetector &detector_sift, cv::SiftDescriptorExtractor &extractor, CodewordUncertainty hard_quant){
    int i = 0;
    for(cv::Mat& sample : samples){
        i++;
        //std::cout << "computing histogram for image " << i << " of " << samples.size() << std::endl;
        Histogram feature_vector;
        compute_histogram(sample, feature_vector, detector_sift, extractor, hard_quant);
        feature_vectors.push_back(feature_vector);
    }
}

//given the feature-space histogram for an image, get the category by finding nearest centroid
int get_category(const Histogram &feature_vector, const std::vector<std::vector<double>> &category_centroids){
    int closest_index = 0;
    double closest_distance = euclidean_distance(category_centroids[0], feature_vector);

    for(int i = 1; i < category_centroids.size(); i++){
        double distance = euclidean_distance(category_centroids[i], feature_vector);
        if(distance < closest_distance){
            closest_index = i;
            closest_distance = distance;
        }
    }

    return closest_index;
}

//categorize images using nearest centroid and generate confusion table
void test_category(std::vector<Histogram> &feature_vectors, std::vector<double> &confusion_table, std::vector<std::string> &category_labels, std::vector<std::vector<double>> &category_centroids){
    for(Histogram &feature_vector : feature_vectors){
        int cat = get_category(feature_vector, category_centroids);
        //std::cout << "image " << i << " of " << bikes.size() << ": " << cat << "-" << category_labels[cat] << std::endl;
        confusion_table[cat]++;
    }

    for(int i = 0; i < confusion_table.size(); i++){
        std::cout << category_labels[i] << " " << confusion_table[i] << ", " << (confusion_table[i]/feature_vectors.size()) << std::endl;
    }
}

int main(int argc, char **argv){

    //load test images -- TODO: may want to make some effort to generalize this to other datasets
    std::cout << "Load Test Images" << std::endl;
    std::vector<std::vector<cv::Mat>> test_images;
    std::vector<std::string> test_labels;
    load_graz2_test(test_images, test_labels);

    //load codebook
    std::cout << "Load Codebook" << std::endl;
    std::vector<std::vector<double>> codebook;
    LoadCodebook("codebook_graz2_800.out", codebook);

    //load nearest centroid classifier
    std::cout << "Load Classifier" << std::endl;
    std::vector<std::string> category_labels;
    std::vector<std::vector<double>> category_centroids;
    load_classifier("graz2_centroid_classifier_800.out", category_labels, category_centroids);

    //TODO: vary these choices to compare performance
    cv::SiftFeatureDetector detector_sift(200); //sift-200 keypoints
    cv::SiftDescriptorExtractor extractor;      //sift128 descriptor
    CodewordUncertainty hard_quant(codebook, 100.0);   //soft quantization

    for(int i = 0; i < test_images.size(); i++){
        std::cout << "Compute Vectors for " << test_labels[i] << std::endl;
        std::vector<Histogram> feature_space_vectors;
        compute_histograms(test_images[i], feature_space_vectors, detector_sift, extractor, hard_quant);

        std::cout << "Categorize " << test_labels[i] << std::endl;
        std::vector<double> confusion_table(category_centroids.size());
        test_category(feature_space_vectors, confusion_table, category_labels, category_centroids);
    }

    return 0;
}
