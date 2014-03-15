#include "Codewords.hpp"
#include <fstream>
#include "../Util/Clustering.hpp"

void LocalDescriptorAndBagOfFeature::FindCodewords(const std::vector<std::vector<double> > &features, int numCodeWords, std::vector<std::vector<double> > &codewords)
{
    std::vector<int> labels, sizes;
    kmeans(features, numCodeWords, labels, codewords, sizes, 10);
}

void LocalDescriptorAndBagOfFeature::FindCodewords(const std::vector<std::vector<double> > &features, int numCodeWords, std::vector<std::vector<double> > &codewords, int iterationCap, int trials)
{
    std::vector<int> labels, sizes;
    int compactness = kmeans(features, numCodeWords, labels, codewords, sizes, iterationCap, trials);
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
        codebook.push_back(codeword);
    }
    filein.close();
}
