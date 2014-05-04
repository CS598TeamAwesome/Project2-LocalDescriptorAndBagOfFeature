#include "Codewords.hpp"
#include <fstream>
#include "../Util/Clustering.hpp"

void LocalDescriptorAndBagOfFeature::FindCodewords(std::vector<std::vector<double> > &features, int numCodeWords, std::vector<std::vector<double> > &codewords)
{
    std::vector<int> labels, sizes;
    kmeans(features, numCodeWords, labels, codewords, sizes, 10, 50);
}

void LocalDescriptorAndBagOfFeature::FindCodewords(std::vector<std::vector<double> > &features, int numCodeWords, std::vector<std::vector<double> > &codewords, int iterationCap, int epsilon, int trials)
{
    std::vector<int> labels, sizes;
    int compactness = kmeans(features, numCodeWords, labels, codewords, sizes, iterationCap, epsilon, trials);
    std::cout << "compactness for kmeans: " << compactness << std::endl;
}

void LocalDescriptorAndBagOfFeature::SaveCodebook(std::string filename, const std::vector<std::vector<double>> &codebook){
    std::ofstream fileout (filename);
    fileout << codebook.size() << std::endl;
    for(const std::vector<double>& code_vector : codebook){
         for(const double& d : code_vector){
             fileout << d << " ";
         }
         fileout << std::endl;
    }
    fileout.close();
}

void LocalDescriptorAndBagOfFeature::LoadCodebook(std::string filename, std::vector<std::vector<double>> &codebook){
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

        std::vector<double> codeword;
        double d;
        while(sin >> d){
            codeword.push_back(d);
        }
        if(codeword.size() == 0){
            std::cout << "empty codeword..." << std::endl;
        } else
            codebook.push_back(codeword);
    }
    filein.close();
}

void LocalDescriptorAndBagOfFeature::SaveVocabularyTree(std::ofstream &fileout, const tree_node &root, int K, int L){
    if(L == 0){
        return;
    }

    for(const tree_node& child : root.children){
         fileout << L << " ";
         for(const double& d : child.value){
             fileout << d << " ";
         }
         fileout << std::endl;
         SaveVocabularyTree(fileout, child, K, L-1);
    }
}

void LocalDescriptorAndBagOfFeature::SaveVocabularyTree(std::string filename, const vocabulary_tree &tree){
    std::ofstream fileout (filename);
    fileout << tree.K << " " << tree.L << std::endl;
    SaveVocabularyTree(fileout, tree.root, tree.K, tree.L);
    fileout.close();
}

void LocalDescriptorAndBagOfFeature::LoadVocabularyTree(std::ifstream &filein, tree_node &root, int K, int L){
    if(L == 0){
        return;
    }

    for(int i = 0; i < K; i++){
        std::string s;
        int pos = filein.tellg(); //in case we need to under the getline
        std::getline(filein, s);
        std::istringstream sin(s);

        int level;
        sin >> level;

        if(L != level){
            //undo the getline and return
            filein.seekg(pos);
            return;
        }

        tree_node child;
        std::vector<double> codeword;
        double d;
        while(sin >> d){
            codeword.push_back(d);
        }
        child.value = codeword;

        LoadVocabularyTree(filein, child, K, L-1);

        root.children.push_back(child);
    }
}

void LocalDescriptorAndBagOfFeature::LoadVocabularyTree(std::string filename, vocabulary_tree &tree){
    //load vocabulary tree file
    std::ifstream filein (filename);
    std::string s;
    std::getline(filein, s);
    std::istringstream sin(s);

    int K, L;
    sin >> K;
    sin >> L;

    tree_node root;
    LoadVocabularyTree(filein, root, K, L);

    tree.root = root;
    tree.K = K;
    tree.L = L;

    filein.close();
}

void LocalDescriptorAndBagOfFeature::FlattenTree(const vocabulary_tree &tree, std::vector<std::vector<double>> &codewords){
        std::vector<tree_node> to_visit;
        std::vector<std::pair<int, int>> to_visit_info; //corresponding {level, index} pairs
        std::vector<std::pair<int, int>> path; //path of {level, index} pairs
        codewords.clear();
        codewords.resize(std::pow(tree.K, tree.L));

        to_visit.push_back(tree.root);
        std::pair<int, int> level_index_pair;
        level_index_pair.first = 0;
        level_index_pair.second = 0;
        to_visit_info.push_back(level_index_pair);

        while(!to_visit.empty()){
            tree_node current = to_visit.back();
            to_visit.pop_back();
            std::pair<int, int> current_info = to_visit_info.back();
            to_visit_info.pop_back();

            if(path.empty()){
                path.push_back(current_info);
            } else {
                while(!path.empty() && path.back().first >= current_info.first){
                    path.pop_back();
                }
                path.push_back(current_info);
            }

            if(current.children.empty()){
                if(current_info.first < tree.L){
                    std::cout << "POTENTIAL PROBLEM - missing intermediate node???" << std::endl;
                }

                int index = 0;
                for(int i = path.size()-1; i > 0; i--){
                    index *= tree.K;
                    index += path[i].second;
                }
                codewords[index] = current.value;
/*
                for(std::pair<int, int> &p : path){
                    std::cout << p.first << "," << p.second << " ";
                }
                std::cout << "= " << index << std::endl;
*/
            }

            for(int i = 0; i < current.children.size(); i++){
                to_visit.push_back(current.children[i]);
                std::pair<int, int> new_info;
                new_info.first = current_info.first + 1;
                new_info.second = i;
                to_visit_info.push_back(new_info);
            }
        }
}
