#pragma once
#include <opencv2/opencv.hpp>
#include <exception>
#include <vector>
#include <array>
#include <functional>
#include <assert.h>
#include <stdlib.h>
#include <numeric>
#include "Distances.hpp"

namespace LocalDescriptorAndBagOfFeature {

    struct tree_node
    {
       std::vector<double> value;
       std::vector<tree_node> children;
    };

    struct vocabulary_tree
    {
        tree_node root;
        int K;
        int L;
    };

    //single run with randomized initial centers
    double kmeans(std::vector<std::vector<double>> input, int K, std::vector<int> &labels, std::vector<std::vector<double>> &centers, std::vector<int> &sizes, int iteration_bound, int epsilon);
    //multiple run, returning the best
    double kmeans(const std::vector<std::vector<double>> &input, int K, std::vector<int> &labels, std::vector<std::vector<double>> &centers, std::vector<int> &sizes, int iteration_bound, int epsilon, int trials);
    //hierarchical, K is the branching factor, L is the number of levels
    void hierarchical_kmeans(const std::vector<std::vector<double>> &input, vocabulary_tree &tree);
    void hierarchical_kmeans(const std::vector<std::vector<double>> &input, int K, int L, tree_node &root);
}
