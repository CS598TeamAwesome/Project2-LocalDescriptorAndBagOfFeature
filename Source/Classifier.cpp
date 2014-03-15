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
#include "Quantization/HardAssignment.hpp"
#include "Quantization/CodewordUncertainty.hpp"
#include "Quantization/Quantization.hpp"
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

//gets centroid for category from training images
void train_category(const std::vector<cv::Mat> &samples, Histogram &centroid, const cv::SiftFeatureDetector &detector_sift, const cv::SiftDescriptorExtractor &extractor, Quantization *quant){
    clock_t start = clock();
    int i = 0;
    for(const cv::Mat& sample : samples){
        i++;
        std::cout << "converting img " << i << " of " << samples.size() << " to bag of features" << std::endl;

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
        Histogram feature_vector;
        quant->quantize(unquantized_features, feature_vector);

        //aggregate
        vector_add(centroid, feature_vector);
    }

    //divide by training category size to compute centroid
    //std::transform(centroid.begin(), centroid.end(), centroid.begin(), std::bind1st(std::divides<double>(),bikes.size()));
    for(double& d : centroid){
        d = d/samples.size();
    }
    std::cout << double( clock() - start ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;
}

void SaveClassifier(std::string filename, const std::vector<std::vector<double>> &category_centroids, const std::vector<std::string> &category_labels){
    std::ofstream fileout (filename);
    fileout << category_centroids.size() << std::endl;
    for(int i = 0; i < category_centroids.size(); i++){
        fileout << category_labels[i] << std::endl;
        for(const double& d : category_centroids[i]){
            fileout << d << " ";
        }
        fileout << std::endl;
    }
    fileout.close();
}

int main(int argc, char **argv){

    //Load training images
    std::cout << "Load Training Images" << std::endl;
    std::vector<std::vector<cv::Mat>> training_images;
    std::vector<std::string> category_labels;
    load_graz2_train(training_images, category_labels);

    //Load codebook
    std::cout << "Load Codebook" << std::endl;
    std::vector<std::vector<double>> codebook;
    LoadCodebook("codebook_graz2_800.out", codebook);

    //Train nearest centroid classifier
    std::vector<std::vector<double>> category_centroids;

    //TODO: vary these choices to compare performance
    cv::SiftFeatureDetector detector_sift(200); //sift-200 keypoints
    cv::SiftDescriptorExtractor extractor;      //sift128 descriptor

    HardAssignment hard_quant(codebook);
    CodewordUncertainty soft_quant(codebook, 100.0);

    Quantization *quant = &soft_quant; //use soft quantization

    for(int i = 0; i < training_images.size(); i++){
        std::cout << "Training " << category_labels[i] << std::endl;
        Histogram centroid(codebook.size());
        train_category(training_images[i],centroid, detector_sift, extractor, quant);
        category_centroids.push_back(centroid);
    }

    //write classifier to file
    SaveClassifier("graz2_centroid_classifier_800.out", category_centroids, category_labels);

    return 0;
}
