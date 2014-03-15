#include "Datasets.hpp"
#include <string>

//hard assumptions on file numbering, tight coupling with labels
void load_graz2_category(std::vector<cv::Mat> &images, std::string path_prefix, int start, int end){
    for(int i = start; i <= end; i++){
        //looping over fixed directory path, with expected file names 0.jpg, 1.jpg, etc
        std::ostringstream convert;
        if(i < 10)
            convert << path_prefix << "00" << i << ".bmp";
        else if(i < 100)
            convert << path_prefix << "0" << i << ".bmp";
        else
            convert << path_prefix << i << ".bmp";

        std::string s = convert.str();
        cv::Mat img = cv::imread(s);
        images.push_back(img);
    }
}

void LocalDescriptorAndBagOfFeature::load_graz2_train(std::vector<std::vector<cv::Mat>> &images, std::vector<std::string> &labels){
    std::array<std::string, 4> path_prefixes = {"Train/bike/bike_", "Train/cars/carsgraz_", "Train/none/bg_graz_", "Train/person/person_"};
    std::array<int, 4> start_indexes = {1, 1, 1, 1};
    std::array<int, 4> end_indexes = {165, 220, 180, 111};

    labels.push_back("Bikes");
    labels.push_back("Cars");
    labels.push_back("Backgrounds");
    labels.push_back("People");

    for(int i = 0; i < 4; i++){
        std::vector<cv::Mat> category_images;
        load_graz2_category(category_images, path_prefixes[i], start_indexes[i], end_indexes[i]);
        images.push_back(category_images);
    }
}

void LocalDescriptorAndBagOfFeature::load_graz2_validate(std::vector<std::vector<cv::Mat>> &images, std::vector<std::string> &labels){
    std::array<std::string, 4> path_prefixes = {"Validation/bike/bike_", "Validation/cars/carsgraz_", "Validation/none/bg_graz_", "Validation/person/person_"};
    std::array<int, 4> start_indexes = {266, 221, 181, 112};
    std::array<int, 4> end_indexes = {365, 320, 280, 211};

    labels.push_back("Bikes");
    labels.push_back("Cars");
    labels.push_back("Backgrounds");
    labels.push_back("People");

    for(int i = 0; i < 4; i++){
        std::vector<cv::Mat> category_images;
        load_graz2_category(category_images, path_prefixes[i], start_indexes[i], end_indexes[i]);
        images.push_back(category_images);
    }
}

void LocalDescriptorAndBagOfFeature::load_graz2_test(std::vector<std::vector<cv::Mat>> &images, std::vector<std::string> &labels){
    std::array<std::string, 4> path_prefixes = {"Test/bike/bike_", "Test/cars/carsgraz_", "Test/none/bg_graz_", "Test/person/person_"};
    std::array<int, 4> start_indexes = {166, 321, 281, 212};
    std::array<int, 4> end_indexes = {265, 420, 380, 311};

    labels.push_back("Bikes");
    labels.push_back("Cars");
    labels.push_back("Backgrounds");
    labels.push_back("People");

    for(int i = 0; i < 4; i++){
        std::vector<cv::Mat> category_images;
        load_graz2_category(category_images, path_prefixes[i], start_indexes[i], end_indexes[i]);
        images.push_back(category_images);
    }
}

const std::array<std::string, 15> scene15_categories = {
        "bedroom",
        "CALsuburb",
        "industrial",
        "kitchen",
        "livingroom",
        "MITcoast",
        "MITforest",
        "MIThighway",
        "MITinsidecity",
        "MITmountain",
        "MITopencountry",
        "MITstreet",
        "MITtallbuilding",
        "PARoffice",
        "store"
};

//hard assumptions on file numbering, tight coupling with labels
void load_scene15_category(std::vector<cv::Mat> &images, std::string category, int start, int end){
    std::string root_folder = "Scene15/";
    for(int i = start; i <= end; i++){
        std::ostringstream convert;
        if(i < 10)
            convert << root_folder << category << "/image_000" << i << ".jpg";
        else if(i < 100)
            convert << root_folder << category << "/image_00" << i << ".jpg";
        else if(i < 1000)
            convert << root_folder << category << "/image_0" << i << ".jpg";
        else
            convert << root_folder << category << "/image_" << i << ".jpg";

        std::string s = convert.str();
        cv::Mat img = cv::imread(s);
        images.push_back(img);
    }
}

void LocalDescriptorAndBagOfFeature::load_scene15_train(std::vector<std::vector<cv::Mat>> &images, std::vector<std::string> &labels){
    for(const std::string& s : scene15_categories){
        labels.push_back(s);
    }

    for(int i = 0; i < 15; i++){
        std::vector<cv::Mat> category_images;
        load_scene15_category(category_images, scene15_categories[i], 1, 50); //TODO: randomize training selection, or pre-divide
        images.push_back(category_images);
    }
}

void LocalDescriptorAndBagOfFeature::load_scene15_test(std::vector<std::vector<cv::Mat>> &images, std::vector<std::string> &labels){
    for(const std::string& s : scene15_categories){
        labels.push_back(s);
    }

    for(int i = 0; i < 15; i++){
        std::vector<cv::Mat> category_images;
        load_scene15_category(category_images, scene15_categories[i], 51, 100); //TODO: select non-training images, or pre-divide
        images.push_back(category_images);
    }
}
