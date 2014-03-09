#pragma once

#include <opencv2/opencv.hpp>
using namespace cv;

namespace LocalDescriptorAndBagOfFeature
{
	class Dog
	{
	public:
		Mat ComputeDog( Mat &img );
	};
}
