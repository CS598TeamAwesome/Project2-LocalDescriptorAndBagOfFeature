#include "NearestCentroidClassifier.hpp"
#include "../Util/Distances.hpp"
#include <fstream>
#include <sstream>
#include <time.h>

//gets centroid for category from training images
void LocalDescriptorAndBagOfFeature::train_category(const std::vector<cv::Mat> &samples, Histogram &centroid, const cv::Ptr<cv::FeatureDetector> &detector, const cv::SiftDescriptorExtractor &extractor, Quantization *quant){
    clock_t start = clock();
    int i = 0;
    for(const cv::Mat& sample : samples){
        i++;
        std::cout << "converting img " << i << " of " << samples.size() << " to bag of features" << std::endl;

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

//given the feature-space histogram for an image, get the category by finding nearest centroid
int LocalDescriptorAndBagOfFeature::get_category(const Histogram &feature_vector, const std::vector<std::vector<double>> &category_centroids){
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
void LocalDescriptorAndBagOfFeature::test_category(std::vector<Histogram> &feature_vectors, std::vector<double> &confusion_table, std::vector<std::string> &category_labels, std::vector<std::vector<double>> &category_centroids){
    for(Histogram &feature_vector : feature_vectors){
        int cat = get_category(feature_vector, category_centroids);
        //std::cout << "image " << i << " of " << bikes.size() << ": " << cat << "-" << category_labels[cat] << std::endl;
        confusion_table[cat]++;
    }

    for(int i = 0; i < confusion_table.size(); i++){
        std::cout << category_labels[i] << " " << confusion_table[i] << ", " << (confusion_table[i]/feature_vectors.size()) << std::endl;
    }
}

void LocalDescriptorAndBagOfFeature::save_classifier(std::string filename, const std::vector<std::vector<double>> &category_centroids, const std::vector<std::string> &category_labels){
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

//gets a centroid classifier trained from test data
void LocalDescriptorAndBagOfFeature::load_classifier(std::string filename, std::vector<std::string> &category_labels, std::vector<std::vector<double>> &category_centroids){
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

        std::vector<double> centroid;
        double d;
        while(sin >> d){
            centroid.push_back(d);
        }
        category_centroids.push_back(centroid);
    }
    filein.close();
}
