#include "VocabularyTreeQuantization.hpp"
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <limits>

using namespace LocalDescriptorAndBagOfFeature;

VocabularyTreeQuantization::VocabularyTreeQuantization(vocabulary_tree &tree):tree(tree){
}

int VocabularyTreeQuantization::get_hierarchical_label(const std::vector<double> &sample, const tree_node &root, int K){
    if(root.children.size() == 0){
        return 0;
    }

    int closest_index = 0;

    double closest_distance = euclidean_distance(root.children[0].value, sample);

    for(int i = 1; i < root.children.size(); i++){
        double distance = euclidean_distance(root.children[i].value, sample);
        if(distance < closest_distance){
            closest_index = i;
            closest_distance = distance;
        }
    }

    return closest_index + K*get_hierarchical_label(sample, root.children[closest_index], K);
}

int VocabularyTreeQuantization::size(){
    return std::pow(tree.K, tree.L);
}

//return the histogram of features for the regions in an image
void VocabularyTreeQuantization::quantize(const std::vector<std::vector<double>> &regions, std::vector<double> &histogram){
    histogram.clear();
    histogram.resize(this->size()); //tree size

    //loop over regions, incrementing the corresponding histogram value for each region
    int index = 0;
    for(const std::vector<double>& region : regions){
        int label = get_hierarchical_label(region, tree.root, tree.K);
        histogram[label]++;
        index++;
    }
}
