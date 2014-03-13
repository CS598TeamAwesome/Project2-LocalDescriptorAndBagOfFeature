#include "DetectFeature.hpp"
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <stdio.h>
#include <iostream>
#include <iterator>
#include <vector>
#include <stdlib.h>
#include <fstream>

using namespace cv;
using namespace std;

Mat DetectFeature::extractFeature (Mat &img){
	Mat img1 = imread(img,CV_LOAD_IMAGE_GRAYSCALE);
	if(!img1.data){
		cout << "Error in reading image" << endl;
	}

	SurfFeatureDetector detector(400);

	vector<KeyPoint> keypoint1;

	detector.detect(img1,keypoint1);

	Mat img_keypoint1;
	drawKeypoints(img1, keypoint1, img_keypoint1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	imshow("Keypoints for image 1", img_keypoint1);

	//waitKey(0);

	return img_keypoint1;
}

