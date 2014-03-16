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
#include "Quantization/HardAssignment.hpp"
#include "Quantization/CodewordUncertainty.hpp"
#include "Quantization/Quantization.hpp"
#include "Util/Datasets.hpp"
#include "Util/Distances.hpp"
#include "Util/Types.hpp"

using std::vector;
using namespace LocalDescriptorAndBagOfFeature;

int main(int argc, char **argv){

    //Load training images
    std::cout << "Load Training Images" << std::endl;
    std::vector<std::vector<cv::Mat>> training_images;
    std::vector<std::string> category_labels;
    load_graz2_train(training_images, category_labels);

    //Load codebook
    std::cout << "Load Codebook" << std::endl;
    std::vector<std::vector<double>> codebook;
    LoadCodebook("codebook_graz2_400_dense.out", codebook);

    //Train nearest centroid classifier
    std::vector<std::vector<double>> category_centroids;

    //TODO: vary these choices to compare performance
    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("Dense");
    //detector->set("featureScaleLevels", 1);
    //detector->set("featureScaleMul", 0.1f);
    //detector->set("initFeatureScale", 1.f);
    detector->set("initXyStep", 30);

    std::vector<std::vector<cv::KeyPoint>> training_keypoints;
    detector->detect( training_images, training_keypoints );


    cv::SiftFeatureDetector detector_sift(200); //sift-200 keypoints
    cv::SiftDescriptorExtractor extractor;      //sift128 descriptor

    HardAssignment hard_quant(codebook);
    CodewordUncertainty soft_quant(codebook, 100.0);

    Quantization *quant = &hard_quant; //use hard quantization

    for(int i = 0; i < training_images.size(); i++){
        std::cout << "Training " << category_labels[i] << std::endl;
        Histogram centroid(codebook.size());
        train_category(training_images[i],centroid, detector, extractor, quant);
        category_centroids.push_back(centroid);
    }

    //write classifier to file
    save_classifier("graz2_centroid_classifier_400_dense.out", category_centroids, category_labels);

    return 0;
}
