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
    double s5[] = {4, 5, 48};

    vector<double> vs1(s1, s1+len);
    vector<double> vs2(s2, s2+len);
    vector<double> vs3(s3, s3+len);
    vector<double> vs4(s4, s4+len);
    vector<double> vs5(s5, s5+len);

    vector<vector<double>> samples;
    samples.push_back(vs1);
    samples.push_back(vs2);
    samples.push_back(vs3);
    samples.push_back(vs4);
    samples.push_back(vs5);

    vector<vector<double>> centers = kmeans(samples, 2);

    //print out cluster centers
    for(int i = 0; i < 2; i++){
        std::cout << "cluster " << i << ": ";
        for(double& d : centers[i]){
            std::cout << d << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}
