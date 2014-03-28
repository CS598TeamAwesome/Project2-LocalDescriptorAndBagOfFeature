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
#include "Quantization/VocabularyTreeQuantization.hpp"
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

    //0. command line arguments
    std::string codebook_filename = "codebook_graz2_200_dense.out";
    //std::string classifier_filename = "classifier-graz2-dense-tree-256.out";
    std::string classifier_filename = "graz2_centroid_classifier_200_dense.out";

    std::string output_filename = "categorization-graz2-dense-200.out";
    std::string quantization_type = "hard";
    std::string detector_type = "Dense";
    std::string descriptor_type = "SIFT";

    std::string error = "Invalid arguments. Usage: [-cl classifier-filename] [-c codebook-filename][-d detector-type][-q quantization-type][-f output-filename]";
    std::string detector_error = "detector-type must be {SIFT, Dense}";
    std::string quant_error = "quantization-type must be {hard, soft}";
    for (int i = 1; i < argc; i++) {
        if (i + 1 != argc){
            std::string s(argv[i]);
            if (s.compare("-f") == 0) {
                output_filename = argv[++i];
            } else if (s.compare("-c") == 0) {
                codebook_filename = argv[++i];
            } else if (s.compare("-cl") == 0) {
                classifier_filename = argv[++i];
            } else if (s.compare("-d") == 0) {
                detector_type = argv[++i];
                if(detector_type.compare("SIFT")!= 0 && detector_type.compare("Dense")!= 0){
                    std::cout << detector_error;
                    return(0);
                }
            } else if (s.compare("-q") == 0) {
                quantization_type = argv[++i];
                if(quantization_type.compare("soft")!= 0 && quantization_type.compare("hard")!= 0){
                    std::cout << quant_error;
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

    std::cout << "Performing Categorization for: detector=" << detector_type << ", descriptor=" << descriptor_type << ", quantization=" << quantization_type << std::endl;

    std::cout << "Load Test Images" << std::endl;
    std::vector<std::vector<cv::Mat>> test_images;
    std::vector<std::string> test_labels;

    load_graz2_test(test_images, test_labels);
    //load_graz2_validate(test_images, test_labels);
    //load_scene15_test(test_images, test_labels);

    //load codebook
    std::cout << "Load Codebook" << std::endl;
    std::vector<std::vector<double>> codebook;
    LoadCodebook(codebook_filename, codebook);

    //load nearest centroid classifier
    std::cout << "Load Classifier" << std::endl;
    std::vector<std::string> category_labels;
    std::vector<std::vector<double>> category_centroids;
    load_classifier(classifier_filename, category_labels, category_centroids);

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

    vocabulary_tree tree;
    LoadVocabularyTree("vocab_tree.out", tree);
    VocabularyTreeQuantization tree_quant(tree);

    Quantization *quant;
    if(quantization_type.compare("hard") == 0){
        quant = &hard_quant;
    } else if(quantization_type.compare("soft") == 0){
        quant = &soft_quant;
    } else if(quantization_type.compare("tree") == 0){
        quant = &tree_quant;
    }

    std::ofstream fileout (output_filename);
    for(int i = 0; i < test_images.size(); i++){
        fileout << "\t" << category_labels[i];
    }
    fileout << std::endl;

    double recall = 0;

    std::vector<std::vector<double>> confusion_table;
    for(int i = 0; i < test_images.size(); i++){
        std::cout << "Compute Vectors for " << test_labels[i] << std::endl;
        std::vector<Histogram> feature_space_vectors;
        compute_histograms(test_images[i], feature_space_vectors, detector, extractor, quant);

        std::cout << "Categorize " << test_labels[i] << std::endl;
        std::vector<double> confusion_row(category_centroids.size());
        test_category(feature_space_vectors, confusion_row, category_labels, category_centroids);

        recall += confusion_row[i]/feature_space_vectors.size();
        std::cout << "recall for category: " << confusion_row[i]/feature_space_vectors.size() << std::endl;

        fileout << category_labels[i];
        //write results to file
        for(int j = 0; j < confusion_row.size(); j++){
            fileout << "\t" << confusion_row[j]/feature_space_vectors.size();
        }
        fileout << std::endl;
        confusion_table.push_back(confusion_row);
    }

    recall /= test_images.size();
    fileout << "Overall Recall: " << recall << std::endl;
    std::cout << "Overall Recall: " << recall << std::endl;

    std::vector<double> predicted_positive_correctly;
    std::vector<double> predicted_positive;
    std::vector<double> actual_positive;
    actual_positive.push_back(100);
    actual_positive.push_back(100);
    actual_positive.push_back(100);
    actual_positive.push_back(100);

    for(int i = 0; i < category_centroids.size(); i++){
        double positives = 0.0;
        for(int j = 0; j < test_images.size(); j++){
            if(i == j){
                predicted_positive_correctly.push_back(confusion_table[i][j]);
            }
            positives += confusion_table[j][i];
        }
        predicted_positive.push_back(positives);
    }

    for(int i = 0; i < test_images.size(); i++){
        std::cout << "cat: " << i << " precision - " << predicted_positive_correctly[i]/predicted_positive[i] << std::endl;
        std::cout << "cat: " << i << " recall - " << predicted_positive_correctly[i]/actual_positive[i] << std::endl;
    }

    fileout.close();;

    return 0;
}
