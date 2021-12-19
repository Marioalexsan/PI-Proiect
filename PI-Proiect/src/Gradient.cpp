#include"Gradient.hpp"

namespace pi {
	cv::Mat calculate_magnitude(cv::Mat Sx, cv::Mat Sy)
	{
		cv::Mat magnitude = cv::Mat(Sx.rows, Sx.cols, CV_64F);
		for (int y = 0; y < Sx.rows; ++y)
		{
			for (int x = 1; x < Sx.cols; ++x)
			{
				double valX_Sx = Sx.at<double>(y, x);
				double valY_Sy = Sy.at<double>(y, x);
				
				magnitude.at<double>(y, x) = sqrt((valX_Sx * valX_Sx) +
					(valY_Sy * valY_Sy));
			}
		}

		return magnitude;
	}
	cv::Mat calculate_orientation(cv::Mat Sx, cv::Mat Sy) 
	{
		cv::Mat orientation = cv::Mat(Sx.rows, Sx.cols, CV_64F);
		for (int y = 0; y < Sx.rows; ++y)
		{
			for (int x = 1; x < Sx.cols; ++x)
			{
				double valX = Sx.at<double>(y, x);
				double valY = Sy.at<double>(y, x);
				//calculez unghiul theta
				orientation.at<double>(y, x) = cv::fastAtan2(valX, valY);
			}
		}

		return orientation;
	}

	void contour_gradient(cv::Mat& image, int dimension)
	{
		cv::Mat Sy, Sx, output; //ERR: daca rows si cols nu sunt egale
		//cv::Mat region = cv::Mat::zeros(dimension, dimension, CV_64F);
		cv::Mat magnitude = cv::Mat::zeros(image.rows, image.cols, CV_64F);
		cv::Mat orientation = cv::Mat::zeros(image.rows, image.cols, CV_64F);

		int cell_width = (int)ceil((double)image.cols / dimension);
		int cell_height = (int)ceil((double)image.rows / dimension);

		//calculam derivatele in directiile x si y
		cv::filter2D(image, Sx, -1, pi::Fx3x3);
		cv::filter2D(image, Sy, -1, pi::Fy3x3);

		//calculez magnitudinea si orientatia (theta)
		magnitude = calculate_magnitude(Sx, Sy);
		orientation = calculate_orientation(Sx, Sy);

		cv::imshow("magnitude", magnitude);
	}
}

