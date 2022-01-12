#include"Project_Headers.hpp"
#include "Constants.hpp"


namespace pi {
	const cv::Mat gauss3x3 = cv::Mat_<double>(
		{
			1, 2, 1,
			2, 4, 2,
			1, 2, 1
		}
	).reshape(0, 3) * (1.0 / 16.0);

	const cv::Mat gauss5x5 = cv::Mat_<double>(
		{
			2, 4, 5, 4, 2,
			4, 9, 12, 9, 4,
			5, 12, 15, 12, 5,
			4, 9, 12, 9, 4,
			2, 4, 5, 4, 2
		}
	).reshape(0, 5) * (1.0 / 159.0);

	const cv::Mat average3x3 = cv::Mat_<double>(
		{
			1, 1, 1,
			1, 1, 1,
			1, 1, 1
		}
	).reshape(0, 3) * (1.0 / 9.0);

	const cv::Mat sharpen3x3 = cv::Mat_<double>(
		{
			-1, -1, -1,
			-1, 9, -1,
			-1, -1, -1
		}
	).reshape(0, 3) * (1.0 / 9.0);

	std::unordered_map<char, cv::Rect> loadLetterRectangles(std::string path) {
		std::ifstream file(path);

		if (!file.good()) {
			std::cout << "Failed to open file " << path << std::endl;
			return std::unordered_map<char, cv::Rect>();
		}

		auto result = std::unordered_map<char, cv::Rect>();

		char ch;
		int x, y, width, height;

		try {
			while (!file.eof()) {
				file >> ch >> x >> y >> width >> height;
				result[ch] = cv::Rect(x, y, width, height);
			}
		}
		catch (...) {
			std::cout << "Failed to read file " << path << std::endl;
			return std::unordered_map<char, cv::Rect>();
		}

		return result;
	}

	const cv::Mat Fx3x3 = cv::Mat_<double>(
	{
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1
	}).reshape(0, 3) / 2.0;

	const cv::Mat Fy3x3 = cv::Mat_<double>(
	{
		-1, -2, -1,
		0, 0, 0,
		1, 2, 1
	}).reshape(0, 3) / 2.0;
}