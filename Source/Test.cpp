#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include "Util/Clustering.hpp"
#include <cmath>
using std::vector;
using namespace LocalDescriptorAndBagOfFeature;

int main(int argc, char **argv){

    int len = 3;

    double s1[] = {0, 1, 8};
    double s2[] = {4, 5, 18};
    double s3[] = {2, 3, 28};
    double s4[] = {3, 4, 38};
    double s5[] = {11, 2, 22};
    double s6[] = {29, 5, 33};
    double s7[] = {13, 50, 19};
    double s8[] = {7, 40, 48};

    vector<double> vs1(s1, s1+len);
    vector<double> vs2(s2, s2+len);
    vector<double> vs3(s3, s3+len);
    vector<double> vs4(s4, s4+len);
    vector<double> vs5(s5, s5+len);
    vector<double> vs6(s6, s6+len);
    vector<double> vs7(s7, s7+len);
    vector<double> vs8(s8, s8+len);

    vector<vector<double>> samples;
    samples.push_back(vs1);
    samples.push_back(vs2);
    samples.push_back(vs3);
    samples.push_back(vs4);
    samples.push_back(vs5);
    samples.push_back(vs6);
    samples.push_back(vs7);
    samples.push_back(vs8);

    vector<int> labels;
    vector<vector<double>> centers;
    vector<int> sizes;

    double compactness = kmeans(samples, 5, labels, centers, sizes, 100);
    std::cout << compactness << std::endl;

    //print out bin labels for samples
    for(int& i : labels){
        std::cout << i << " ";
    }
    std::cout << std::endl;

    //print out cluster centers
    for(int i = 0; i < 5; i++){
        std::cout << "cluster " << i << "(" << sizes[i] << ")" << ": ";
        for(double& d : centers[i]){
            std::cout << d << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}
