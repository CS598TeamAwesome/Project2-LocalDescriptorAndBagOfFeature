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
#include "Util/Distances.hpp"
#include "Util/Types.hpp"

using std::vector;
using namespace LocalDescriptorAndBagOfFeature;

void load_bikes(std::vector<cv::Mat> &images){
    std::cout << "Loading Bikes" << std::endl;

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
}

void load_cars(std::vector<cv::Mat> &images){
    std::cout << "Loading Cars" << std::endl;

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
}

void load_background(std::vector<cv::Mat> &images){
    std::cout << "Loading Backgrounds" << std::endl;

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
}

void load_people(std::vector<cv::Mat> &images){
    std::cout << "Loading People" << std::endl;

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
}

void convert_mat_to_vector(const cv::Mat &descriptors, std::vector<std::vector<double>> &samples){
    for(int i = 0; i < descriptors.rows; i++){
        //Mat to vector<double> conversion method as described in the OpenCV Documentation
        const double* p = descriptors.ptr<double>(i);
        std::vector<double> vec(p, p + descriptors.cols);
        samples.push_back(vec);
    }
}

//gets centroid for category from training images
void train_category(const std::vector<cv::Mat> &samples, Histogram &centroid, const cv::SiftFeatureDetector &detector_sift, const cv::SiftDescriptorExtractor &extractor, HardAssignment &hard_quant){
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
        hard_quant.quantize(unquantized_features, feature_vector);

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

    //Load training images -- TODO: may want to make some effort to generalize this to other datasets
    std::vector<cv::Mat> bikes, cars, backgrounds, people;
    load_bikes(bikes);
    load_cars(cars);
    load_background(backgrounds);
    load_people(people);

    std::vector<std::vector<cv::Mat>> training_images;
    training_images.push_back(bikes);
    training_images.push_back(cars);
    training_images.push_back(backgrounds);
    training_images.push_back(people);

    std::vector<std::string> category_labels;
    category_labels.push_back("Bikes");
    category_labels.push_back("Cars");
    category_labels.push_back("Backgrounds");
    category_labels.push_back("People");

    //Load codebook
    std::cout << "Load Codebook" << std::endl;
    std::vector<std::vector<double>> codebook;
    LoadCodebook("codebook_graz2_25_5.out", codebook);

    //Train nearest centroid classifier
    std::vector<std::vector<double>> category_centroids;

    //TODO: vary these choices to compare performance
    cv::SiftFeatureDetector detector_sift(200); //sift-200 keypoints
    cv::SiftDescriptorExtractor extractor;      //sift128 descriptor
    HardAssignment hard_quant(codebook);        //hard quantization

    for(int i = 0; i < training_images.size(); i++){
        std::cout << "Training " << category_labels[i] << std::endl;
        Histogram centroid(codebook.size());
        train_category(training_images[i],centroid, detector_sift, extractor, hard_quant);
        category_centroids.push_back(centroid);
    }

    //write classifier to file
    SaveClassifier("graz2_centroid_classifier_25_5.out", category_centroids, category_labels);

    return 0;
}
