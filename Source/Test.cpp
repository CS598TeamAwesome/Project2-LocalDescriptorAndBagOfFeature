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
#include "SVM/svm.h"
#include "Util/Datasets.hpp"
#include "Util/Distances.hpp"
#include "Util/Types.hpp"

using std::vector;
using namespace LocalDescriptorAndBagOfFeature;

void save_problem(std::string filename, const svm_problem &prob, int codebook_size);

void compute_bow_histogram(cv::Mat &sample, Histogram &feature_vector, cv::Ptr<cv::FeatureDetector> &detector, cv::SiftDescriptorExtractor &extractor, Quantization *quant);
void compute_bow_histograms(std::vector<cv::Mat> &samples, std::vector<Histogram> &feature_vectors, cv::Ptr<cv::FeatureDetector> &detector, cv::SiftDescriptorExtractor &extractor, Quantization *quant);
void compute_bow_histograms(std::vector<cv::Mat> &samples, std::vector<Histogram> &feature_vectors);

int main(int argc, char **argv){
    std::string train_filename = "bikes_train";
    int histogram_size = 200; //size of histogram representation (vocab size for bag of words, color space for color hist, etc.)
    int dataset_size = 676; //676 for graz2 train, 400 for graz2 validation and test
    int positive_category = 0; //the category we are training for
    //for graz2, 0 - bikes, 1 - cars, 2 - background, 3 - people

    //Load training images
    std::cout << "Load Training Images" << std::endl;
    std::vector<std::vector<cv::Mat>> training_images;
    std::vector<std::string> category_labels;
    load_graz2_train(training_images, category_labels);
    //load_graz2_validate(training_images, category_labels);
    //load_graz2_test(training_images, category_labels);
    //load_scene15_train(training_images, category_labels);


    //build Problem for SVM
    svm_problem prob;
    prob.l = dataset_size;
    prob.y = new double[dataset_size];
    prob.x = new svm_node*[dataset_size];

    int index = 0;
    //a. compute histograms for images
    for(int i = 0; i < training_images.size(); i++){
        std::cout << "Compute Vectors for " << category_labels[i] << std::endl;
        std::vector<Histogram> feature_space_vectors;
        compute_bow_histograms(training_images[i], feature_space_vectors);
        //compute_color_histogram()
        //compute_histogram_of_oriented_gradient()

        std::cout << "converting vectors into svm_problem" << std::endl;
        //b. convert histograms into svm_problem, including implicit label (we choose Bike +1, other -1)
        for(int j = 0; j < feature_space_vectors.size(); j++){
            if(i == positive_category){
                prob.y[index] = 1;
            } else {
                prob.y[index] = -1;
            }

            svm_node *hist = new svm_node[histogram_size]; //hist size is number of words in vocabulary
            for(int k = 0; k < histogram_size; k++){
                svm_node item;
                item.index = k+1;
                item.value = feature_space_vectors[j][k];
                hist[k] = item;
            }
            prob.x[index] = hist;
            index++;
        }
    }

    //write problem to file
    save_problem(train_filename, prob, histogram_size);

    return 0;
}

//this produces the file we can feed into svm-train
void save_problem(std::string filename, const svm_problem &prob, int codebook_size){
    std::ofstream fileout (filename);
    for(int i = 0; i < prob.l; i++){
        fileout << prob.y[i] << " ";
        for(int j = 0; j < codebook_size; j++){
            fileout << prob.x[i][j].index << ":" << prob.x[i][j].value << " ";
        }
        fileout << std::endl;
    }
    fileout.close();
}

//using bag of visual words as the image representation (vs. color histogram, hog, etc.)
void compute_bow_histograms(std::vector<cv::Mat> &samples, std::vector<Histogram> &feature_vectors){

    //0. settings
    std::string codebook_filename = "codebook_graz2_200_dense.out";
    std::string quantization_type = "hard";
    std::string detector_type = "Dense";
    std::string descriptor_type = "SIFT";

    //Load codebook
    std::cout << "Load Codebook" << std::endl;
    std::vector<std::vector<double>> codebook;
    LoadCodebook(codebook_filename, codebook);

    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(detector_type);
    if(detector_type.compare("Dense") == 0){
        //detector->set("featureScaleLevels", 1);
        //detector->set("featureScaleMul", 0.1f);
        //detector->set("initFeatureScale", 1.f);
        detector->set("initXyStep", 30); //15 for scene15, 30 for graz2
    } else if(detector_type.compare("SIFT") == 0){
        detector->set("nFeatures", 200);
    }

    cv::SiftDescriptorExtractor extractor; //sift128 descriptor

    HardAssignment hard_quant(codebook);
    CodewordUncertainty soft_quant(codebook, 100.0); //default smoothing value 100.0
    Quantization *quant;
    if(quantization_type.compare("hard") == 0){
        quant = &hard_quant;
    } else if(quantization_type.compare("soft") == 0){
        quant = &soft_quant;
    }

    compute_bow_histograms(samples, feature_vectors, detector, extractor, quant);
}

void compute_bow_histogram(cv::Mat &sample, Histogram &feature_vector, cv::Ptr<cv::FeatureDetector> &detector, cv::SiftDescriptorExtractor &extractor, Quantization *quant){
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

void compute_bow_histograms(std::vector<cv::Mat> &samples, std::vector<Histogram> &feature_vectors, cv::Ptr<cv::FeatureDetector> &detector, cv::SiftDescriptorExtractor &extractor, Quantization *quant){
    int i = 0;
    for(cv::Mat& sample : samples){
        i++;
        std::cout << "computing histogram for image " << i << " of " << samples.size() << std::endl;
        Histogram feature_vector;
        compute_bow_histogram(sample, feature_vector, detector, extractor, quant);
        feature_vectors.push_back(feature_vector);
    }
}
