#include "DetectFeature.hpp"

using namespace LocalDescriptorAndBagOfFeature;

Mat DetectFeature::extractFeature (Mat &img)
{
	if(!img.data){
		cout << "Error in reading image" << endl;
	}

	Mat img_gray;
	cvtColor( img, img_gray, CV_RGB2GRAY );

	SurfFeatureDetector detector(400);

	vector<KeyPoint> keypoint1;

	detector.detect(img_gray,keypoint1);

	Mat img_keypoint1;
	drawKeypoints(img_gray, keypoint1, img_keypoint1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	//imshow("Keypoints for image 1", img_keypoint1);

	return img_keypoint1;
}

