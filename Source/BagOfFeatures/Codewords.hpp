#pragma once
#include <vector>
#include <string>
#include "../Util/Clustering.hpp"

namespace LocalDescriptorAndBagOfFeature 
{
    void FindCodewords(std::vector<std::vector<double>> &features, int numCodeWords, std::vector<std::vector<double>> &codewords);
    void FindCodewords(std::vector<std::vector<double>> &features, int numCodeWords, std::vector<std::vector<double>> &codewords, int iterationCap, int epsilon, int trials);
    void SaveCodebook(std::string filename, const std::vector<std::vector<double>> &codebook);
    void LoadCodebook(std::string filename, std::vector<std::vector<double>> &codebook);

    void SaveVocabularyTree(std::ofstream &fileout, const tree_node &root, int K, int L);
    void SaveVocabularyTree(std::string filename, const vocabulary_tree &tree);
    void LoadVocabularyTree(std::string filename, vocabulary_tree &tree);
    void LoadVocabularyTree(std::ifstream &filein, tree_node &root, int K, int L);

    void FlattenTree(const vocabulary_tree &tree, std::vector<std::vector<double>> &codewords);
}
