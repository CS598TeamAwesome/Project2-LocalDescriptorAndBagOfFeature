#include "Dog.hpp"

using namespace LocalDescriptorAndBagOfFeature;

Mat Dog::ComputeDog( Mat &img )
{
	Mat img_gray;
	cvtColor( img, img_gray, CV_RGB2GRAY );

	Mat img_1, img_2, img_3, img_4, img_5;
	Mat img_out = img;

	// set the filter to 7 X 7
	Size k;
	k.width = 7;
	k.height = 7;

	// create 5 blur image
	GaussianBlur( img_gray, img_1, k, 1.0, 1.0, BORDER_DEFAULT );
	GaussianBlur( img_gray, img_2, k, 2.0, 2.0, BORDER_DEFAULT );
	GaussianBlur( img_gray, img_3, k, 4.0, 4.0, BORDER_DEFAULT );
	GaussianBlur( img_gray, img_4, k, 8.0, 8.0, BORDER_DEFAULT );
	GaussianBlur( img_gray, img_5, k, 16.0, 16.0, BORDER_DEFAULT );

	for ( int i = 0; i < img.cols ; i++ )
	{
		for ( int j = 0; j < img.rows; j++ )
		{
			int temp1 = img_1.data[j*img_1.cols + i] - img_2.data[j*img_1.cols + i];
			int temp2 = img_2.data[j*img_1.cols + i] - img_3.data[j*img_1.cols + i];
			int temp3 = img_3.data[j*img_1.cols + i] - img_4.data[j*img_1.cols + i];
			int temp4 = img_4.data[j*img_1.cols + i] - img_5.data[j*img_1.cols + i];

			int temp_s1 = temp1 - temp2;
			int temp_s2 = temp3 - temp4;

			int temp = temp_s1 - temp_s2;

			if ( temp > 0 )
			{
				img_out.at<Vec3b>(j,i) = Vec3b( 0, 255, 0 );
			}

			// I keep these two cuz when k and Gaussian value change, the value of edge temp changes
			else if ( temp < 0 )
			{
				//img_out.at<Vec3b>(j,i) = Vec3b( 0, 255, 255 );
			}
			else
			{
				//img_out.at<Vec3b>(j,i) = Vec3b( 255, 0, 255 );
			}
		}
	}

	return img_out;
}
