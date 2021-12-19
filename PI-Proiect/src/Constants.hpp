#pragma once
#include"Project_Headers.hpp"

namespace pi {
	extern const cv::Mat gauss3x3;

	extern const cv::Mat gauss5x5;

	extern const cv::Mat average3x3;

	extern const cv::Mat sharpen3x3;

	std::unordered_map<char, cv::Rect> loadLetterRectangles(std::string path);

	extern const cv::Mat Fx3x3;

	extern const cv::Mat Fy3x3;
}