#pragma once
#include"Project_Headers.hpp"
#include "Constants.hpp"


namespace pi {
	cv::Mat calculate_magnitude(cv::Mat Sx, cv::Mat Sy);
	cv::Mat calculate_orientation(cv::Mat Sx, cv::Mat Sy);
	void contour_gradient(cv::Mat& image, int dimension);
}