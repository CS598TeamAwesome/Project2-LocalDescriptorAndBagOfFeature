#pragma once
#include <opencv2/opencv.hpp>
#include <exception>
#include <vector>
#include <cmath>
#include <array>
#include <functional>
#include <assert.h>
#include <stdlib.h>
#include "Quantization.hpp"
#include "../Util/Distances.hpp"
#include "../Util/Clustering.hpp"

namespace LocalDescriptorAndBagOfFeature {

    class VocabularyTreeQuantization : public Quantization {
        public:
            VocabularyTreeQuantization(vocabulary_tree &tree);
            int get_hierarchical_label(const std::vector<double> &sample, const tree_node &root, int K);
            void quantize(const std::vector<std::vector<double>> &regions, std::vector<double> &histogram);
            int size();

        private:
            vocabulary_tree tree;
    };
}
