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

void load_codebook(std::string filename, std::vector<std::vector<double>> &codebook){
    //load codebook from file
    std::ifstream filein (filename);
    std::string s;
    std::getline(filein, s);
    std::istringstream sin(s);

    int codeword_ct;
    sin >> codeword_ct;

    for(int i = 0; i < codeword_ct; i++){
        getline(filein, s);
        sin.str(s);
        sin.clear();

        vector<double> codeword;
        double d;
        while(sin >> d){
            codeword.push_back(d);
        }
        codebook.push_back(codeword);
    }
    filein.close();
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

int main(int argc, char **argv){

    //load training images -- TODO: may want to make some effort to generalize this to other datasets
    std::vector<cv::Mat> bikes, cars, backgrounds, people;
    load_bikes(bikes);
    load_cars(cars);
    load_background(backgrounds);
    load_people(people);

    //load codebook
    std::cout << "Load Codebook" << std::endl;
    std::vector<std::vector<double>> codebook;
    load_codebook("codebook_graz2.out", codebook);

    //train nearest centroid classifier
    std::vector<std::string> category_labels;
    std::vector<std::vector<double>> category_centroids;

    //TODO: vary these choices to compare performance
    cv::SiftFeatureDetector detector_sift(200); //sift-200 keypoints
    cv::SiftDescriptorExtractor extractor;      //sift128 descriptor
    HardAssignment hard_quant(codebook);        //hard quantization

    //TODO: generalize this -- apply to other datasets

    std::cout << "Training Bikes" << std::endl;
    //train bikes
    Histogram bike_centroid(codebook.size());
    train_category(bikes, bike_centroid, detector_sift, extractor, hard_quant);
    category_labels.push_back("bikes");
    category_centroids.push_back(bike_centroid);

    std::cout << "Training Cars" << std::endl;
    //train cars
    Histogram car_centroid(codebook.size());
    train_category(cars, car_centroid, detector_sift, extractor, hard_quant);
    category_labels.push_back("cars");
    category_centroids.push_back(car_centroid);

    std::cout << "Training Background" << std::endl;
    //train background
    Histogram bg_centroid(codebook.size());
    train_category(backgrounds, bg_centroid, detector_sift, extractor, hard_quant);
    category_labels.push_back("backgrounds");
    category_centroids.push_back(bg_centroid);

    std::cout << "Training People" << std::endl;
    //train people
    Histogram people_centroid(codebook.size());
    train_category(people, people_centroid, detector_sift, extractor, hard_quant);
    category_labels.push_back("people");
    category_centroids.push_back(people_centroid);

    //write classifier to file
    std::ofstream fileout ("graz2_centroid_classifier.out");
    fileout << category_centroids.size() << std::endl;
    for(int i = 0; i < category_centroids.size(); i++){
        fileout << category_labels[i] << std::endl;
        for(double& d : category_centroids[i]){
            fileout << d << " ";
        }
        fileout << std::endl;
    }
    fileout.close();

    return 0;
}
