#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <time.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include "Util/Clustering.hpp"
#include "Util/Distances.hpp"
#include "Util/Datasets.hpp"
#include "Quantization/HardAssignment.hpp"
#include "Quantization/CodewordUncertainty.hpp"
#include "BagOfFeatures/Codewords.hpp"
#include <cmath>
using std::vector;
using namespace LocalDescriptorAndBagOfFeature;

int main(int argc, char **argv){

    //0. read command line arguments or set default
    int vocabulary_size = 100; //number of clusters for k-means
    int iteration_cap = 15; //number of iterations for k-means
    int trials = 5; //number of k-mean trials
    int epsilon = 100; //termination condition for k-means, if between iterations, compactness increases < epsilon, stop

    std::string detector_type = "Dense";
    std::string descriptor_type = "SIFT";
    std::string output_filename = "Codebook_5.out";

    std::string error = "Invalid arguments. Usage: [-f output-filename] [-v vocabulary-size][-d detector-type]";
    std::string detector_error = "detector-type must be {SIFT, Dense}";
    std::string vocab_error = "vocabulary size must be an integer greater than 0";
    for (int i = 1; i < argc; i++) {
        if (i + 1 != argc){
            std::string s(argv[i]);
            if (s.compare("-f") == 0) {
                output_filename = argv[++i];
            } else if (s.compare("-v") == 0) {
                vocabulary_size = std::atoi(argv[++i]);
                if(vocabulary_size <= 0){
                    std::cout << vocab_error;
                    return(0);
                }
            } else if (s.compare("-d") == 0) {
                detector_type = argv[++i];
                if(detector_type.compare("SIFT")!= 0 && detector_type.compare("Dense")!= 0){
                    std::cout << detector_error;
                    return(0);
                }
            } else {
                std::cout << error;
                return(0);
            }
        } else {
            std::cout << error;
            return(0);
        }
    }

    std::cout << "Building Codebook for: vocab-size=" << vocabulary_size << ", detector=" << detector_type << ", descriptor=" << descriptor_type << std::endl;

    //1. load training images
    std::vector<std::vector<cv::Mat>> images_by_category;
    std::vector<std::string> category_labels;

    //load_scene15_train(images_by_category, category_labels);
    load_graz2_train(images_by_category, category_labels);

    //flatten
    std::vector<cv::Mat> training_images;
    for(std::vector<cv::Mat>& cat : images_by_category){
        training_images.insert(training_images.end(), cat.begin(), cat.end());
    }

    //2. detect keypoints
    std::cout << "Detecting Keypoints" << std::endl;
    clock_t start = clock();

    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(detector_type);

    //at the moment user cannot set these from the command line
    if(detector_type.compare("Dense") == 0){
        //detector->set("featureScaleLevels", 1);
        //detector->set("featureScaleMul", 0.1f);
        //detector->set("initFeatureScale", 1.f);
        detector->set("initXyStep", 25); //for graz2, 30 gets ~352 per image, for scene15, 15 gets ~314
    } else if(detector_type.compare("SIFT") == 0){
        detector->set("nFeatures", 200);
    }


    std::vector<std::vector<cv::KeyPoint>> training_keypoints;
    detector->detect( training_images, training_keypoints );

    int size = 0;
    for(std::vector<cv::KeyPoint>& points : training_keypoints){
        size += points.size();
    }

    std::cout << "keypoints: " << size << ", keypoints per image: " << size/training_keypoints.size() << std::endl;
    std::cout << double( clock() - start ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;

    //3. compute descriptors
    std::cout << "Computing Descriptors" << std::endl;
    start = clock();
    cv::SiftDescriptorExtractor extractor;
    std::vector<cv::Mat> training_descriptors;
    for(int i = 0; i < training_images.size(); i++){
        cv::Mat descriptor_uchar;
        extractor.compute(training_images[i], training_keypoints[i], descriptor_uchar);

        cv::Mat descriptor_double;
        descriptor_uchar.convertTo(descriptor_double, CV_64F);

        training_descriptors.push_back(descriptor_double);

        if(i%50 == 0){
            std::cout << "... finished for " << i << std::endl;
        }
    }
    std::cout << double( clock() - start ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;
    std::cout << "training descriptors: " << training_descriptors.size() << std::endl;

    std::vector<std::vector<double>> samples;
    for(const cv::Mat& mat : training_descriptors){
        convert_mat_to_vector(mat, samples);
    }

    //free mats
    for(cv::Mat& mat : training_descriptors){
        mat.release();
    }
    //x. build vocabulary tree
    start = clock();
    std::cout << "Build Vocabulary Tree" << std::endl;
    vocabulary_tree tree;
    tree.K = 5; //branching factor
    tree.L = 4; //depth
    hierarchical_kmeans(samples, tree);
    std::cout << double( clock() - start ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;

    //x2. save to file
    SaveVocabularyTree("vocab_tree_625.out", tree);

    vocabulary_tree loaded_tree;
    LoadVocabularyTree("vocab_tree_625.out", loaded_tree);

    SaveVocabularyTree("vocab_tree_resave_625.out", loaded_tree);
/*
    //4. cluster to codewords
    start = clock();
    std::cout << "Find Codewords" << std::endl;
    vector<vector<double>> centers;
    FindCodewords(samples, vocabulary_size, centers, iteration_cap, epsilon, trials);
    std::cout << double( clock() - start ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;

    //5. write codebook to file
    SaveCodebook(output_filename, centers);
*/
    return 0;
}
