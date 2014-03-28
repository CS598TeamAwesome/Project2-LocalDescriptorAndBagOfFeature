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
#include "Quantization/VocabularyTreeQuantization.hpp"
#include "Util/Datasets.hpp"
#include "Util/Distances.hpp"
#include "Util/Types.hpp"

using std::vector;
using namespace LocalDescriptorAndBagOfFeature;

int main(int argc, char **argv){
    //0. read command line arguments or set default
    std::string codebook_filename = "codebook-scene15-200-dense.out";
    std::string output_filename = "classifier-graz2-dense-tree-625b.out";
    std::string quantization_type = "tree";
    std::string detector_type = "Dense";
    std::string descriptor_type = "SIFT";

    std::string error = "Invalid arguments. Usage: [-f output-filename] [-c codebook-filename][-d detector-type][-q quantization-type]";
    std::string detector_error = "detector-type must be {SIFT, Dense}";
    std::string quant_error = "quantization-type must be {hard, soft}";
    for (int i = 1; i < argc; i++) {
        if (i + 1 != argc){
            std::string s(argv[i]);
            if (s.compare("-f") == 0) {
                output_filename = argv[++i];
            } else if (s.compare("-c") == 0) {
                codebook_filename = argv[++i];
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

    std::cout << "Training Classifier for: detector=" << detector_type << ", descriptor=" << descriptor_type << ", quantization=" << quantization_type << std::endl;

    //Load training images
    std::cout << "Load Training Images" << std::endl;
    std::vector<std::vector<cv::Mat>> training_images;
    std::vector<std::string> category_labels;
    load_graz2_train(training_images, category_labels);
    //load_scene15_train(training_images, category_labels);

    //Load codebook
    std::cout << "Load Codebook" << std::endl;
    std::vector<std::vector<double>> codebook;
    LoadCodebook(codebook_filename, codebook);

    //Train nearest centroid classifier
    std::vector<std::vector<double>> category_centroids;

    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(detector_type);
    if(detector_type.compare("Dense") == 0){
        //detector->set("featureScaleLevels", 1);
        //detector->set("featureScaleMul", 0.1f);
        //detector->set("initFeatureScale", 1.f);
        detector->set("initXyStep", 25); //15 for scene15, 30 for graz2
    } else if(detector_type.compare("SIFT") == 0){
        detector->set("nFeatures", 200);
    }

    cv::SiftDescriptorExtractor extractor; //sift128 descriptor

    HardAssignment hard_quant(codebook);
    CodewordUncertainty soft_quant(codebook, 100.0); //default smoothing value 100.0

    vocabulary_tree tree;
    LoadVocabularyTree("vocab_tree_625.out", tree);
    VocabularyTreeQuantization tree_quant(tree);

    int vocabulary_size = 0;

    Quantization *quant;
    if(quantization_type.compare("hard") == 0){
        quant = &hard_quant;
        vocabulary_size = codebook.size();
    } else if(quantization_type.compare("soft") == 0){
        quant = &soft_quant;
        vocabulary_size = codebook.size();
    } else if(quantization_type.compare("tree") == 0){
        quant = &tree_quant;
        vocabulary_size = tree_quant.size(); //tree size
    }

    for(int i = 0; i < training_images.size(); i++){
        std::cout << "Training " << category_labels[i] << std::endl;
        Histogram centroid(vocabulary_size);
        train_category(training_images[i],centroid, detector, extractor, quant);
        category_centroids.push_back(centroid);
    }

    //write classifier to file
    save_classifier(output_filename, category_centroids, category_labels);

    return 0;
}
