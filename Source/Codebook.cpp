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
#include "Quantization/HardAssignment.hpp"
#include "Quantization/CodewordUncertainty.hpp"
#include "BagOfFeatures/Codewords.hpp"
#include <cmath>
using std::vector;
using namespace LocalDescriptorAndBagOfFeature;

void load_training_images(std::vector<cv::Mat> &images){
    std::cout << "Loading input images" << std::endl;
    clock_t start = clock();

    //GRAZ2 bikes
    for(int i = 1; i <= 165; i++){
        //looping over fixed directory path, with expected file names 0.jpg, 1.jpg, etc
        std::ostringstream convert;
        if(i < 10)
            convert << "Train/bike/bike_00" << i << ".bmp";
        else if(i < 100)
            convert << "Train/bike/bike_0" << i << ".bmp";
        else
            convert << "Train/bike/bike_" << i << ".bmp";

        std::string s = convert.str();
        cv::Mat img = cv::imread(s);
        images.push_back(img);
    }

    std::cout << "... bike images loaded" << std::endl;

    //GRAZ2 cars
    for(int i = 1; i <= 220; i++){
        //looping over fixed directory path, with expected file names 0.jpg, 1.jpg, etc
        std::ostringstream convert;
        if(i < 10)
            convert << "Train/cars/carsgraz_00" << i << ".bmp";
        else if(i < 100)
            convert << "Train/cars/carsgraz_0" << i << ".bmp";
        else
            convert << "Train/cars/carsgraz_" << i << ".bmp";

        std::string s = convert.str();
        cv::Mat img = cv::imread(s);
        images.push_back(img);
    }

    std::cout << "... car images loaded" << std::endl;

    //GRAZ2 none
    for(int i = 1; i <= 180; i++){
        //looping over fixed directory path, with expected file names 0.jpg, 1.jpg, etc
        std::ostringstream convert;
        if(i < 10)
            convert << "Train/none/bg_graz_00" << i << ".bmp";
        else if(i < 100)
            convert << "Train/none/bg_graz_0" << i << ".bmp";
        else
            convert << "Train/none/bg_graz_" << i << ".bmp";

        std::string s = convert.str();
        cv::Mat img = cv::imread(s);
        images.push_back(img);
    }

    std::cout << "... background images loaded" << std::endl;

    //GRAZ2 person
    for(int i = 1; i <= 111; i++){
        //looping over fixed directory path, with expected file names 0.jpg, 1.jpg, etc
        std::ostringstream convert;
        if(i < 10)
            convert << "Train/person/person_00" << i << ".bmp";
        else if(i < 100)
            convert << "Train/person/person_0" << i << ".bmp";
        else
            convert << "Train/person/person_" << i << ".bmp";

        std::string s = convert.str();
        cv::Mat img = cv::imread(s);
        images.push_back(img);
    }

    std::cout << "... people images loaded" << std::endl;

    std::cout << double( clock() - start ) / (double)CLOCKS_PER_SEC<< " seconds. - load images" << std::endl;

}

void convert_mat_to_vector(const std::vector<cv::Mat> &descriptors, std::vector<std::vector<double>> &samples){
    for(const cv::Mat& mat : descriptors){
        for(int i = 0; i < mat.rows; i++){
            //Mat to vector<double> conversion method as described in the OpenCV Documentation
            const double* p = mat.ptr<double>(i);
            std::vector<double> vec(p, p + mat.cols);
            samples.push_back(vec);
        }
    }
}

int main(int argc, char **argv){

    //1. load training images
    std::vector<cv::Mat> training_images;
    load_training_images(training_images);
    std::cout << "Total Count: " << training_images.size() << std::endl;

    //2. detect keypoints
    std::cout << "Detecting Keypoints" << std::endl;
    clock_t start = clock();

    std::vector<std::vector<cv::KeyPoint>> training_keypoints;
    cv::SiftFeatureDetector detector_sift(200);
    detector_sift.detect( training_images, training_keypoints );

    std::cout << "keypoint vectors: " << training_keypoints.size() << std::endl;
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

    std::vector<std::vector<double>> samples;
    convert_mat_to_vector(training_descriptors, samples);

    //4. cluster to codewords
    std::cout << "Perform K-means Clustering" << std::endl;
    start = clock();
    vector<int> labels;
    vector<vector<double>> centers;
    vector<int> sizes;
    double compactness = kmeans(samples, 200, labels, centers, sizes, 30, 1); //single run with random centers
    //double compactness = kmeans(samples, 400, labels, centers, sizes, 30, 1); //400 words - Xiaoran
    //double compactness = kmeans(samples, 800, labels, centers, sizes, 30, 1); //800 words - Miao
    //double compactness = kmeans(samples, 1600, labels, centers, sizes, 30, 1); //1600 words - Cheng
    std::cout << compactness << std::endl;
    std::cout << double( clock() - start ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;

    //5. write codebook to file
    std::ofstream fileout ("codebook_graz2.out");
    fileout << centers.size() << std::endl;
    for(vector<double>& code_vector : centers){
         for(double& d : code_vector){
             fileout << d << " ";
         }
         fileout << std::endl;
    }
    fileout.close();

    return 0;
}
