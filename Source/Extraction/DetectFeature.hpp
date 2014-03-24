#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <stdio.h>
#include <iostream>
#include <iterator>
#include <vector>
#include <stdlib.h>
#include <fstream>

using namespace cv;
using namespace std;

namespace LocalDescriptorAndBagOfFeature
{

    class DetectFeature
	{
		public:
			Mat extractFeature(Mat &img);
	};

}
