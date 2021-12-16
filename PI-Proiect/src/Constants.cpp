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

	const std::unordered_map<char, cv::Rect> letter_sheet = {
			{'A', cv::Rect(1, 1, 32, 37)},
			{'B', cv::Rect(35, 1, 26, 37)},
			{'C', cv::Rect(65, 1, 26, 37)},
			{'D', cv::Rect(94, 1, 27, 37)},
			{'E', cv::Rect(124, 0, 24, 37)},
			{'F', cv::Rect(151, 0, 25, 37)},
			{'G', cv::Rect(179, 0, 26, 37)},
			{'H', cv::Rect(208, 1, 26, 37)},
			{'I', cv::Rect(237, 1, 6, 37)},
			{'J', cv::Rect(246, 1, 22, 37)},
			{'K', cv::Rect(271, 1, 28, 37)},
			{'L', cv::Rect(302, 1, 24, 37)},
			{'M', cv::Rect(328, 1, 32, 37)},
			{'N', cv::Rect(363, 1, 28, 37)},
			{'O', cv::Rect(394, 0, 27, 38)},
			{'P', cv::Rect(423, 1, 26, 37)},
			{'Q', cv::Rect(451, 0, 30, 38)},
			{'R', cv::Rect(483, 1, 27, 37)},
			{'S', cv::Rect(513, 0, 27, 38)},
			{'T', cv::Rect(541, 1, 26, 37)},
			{'U', cv::Rect(568, 1, 26, 37)},
			{'V', cv::Rect(596, 1, 29, 36)},
			{'W', cv::Rect(625, 1, 42, 37)},
			{'X', cv::Rect(667, 1, 29, 37)},
			{'Y', cv::Rect(696, 1, 27, 37)},
			{'Z', cv::Rect(725, 1, 24, 37)},
			{'0', cv::Rect(1, 62, 22, 38)},
			{'1', cv::Rect(28, 63, 11, 37)},
			{'2', cv::Rect(45, 63, 21, 36)},
			{'3', cv::Rect(72, 62, 22, 38)},
			{'4', cv::Rect(100, 63, 24, 37)},
			{'5', cv::Rect(129, 63, 22, 37)},
			{'6', cv::Rect(157, 63, 21, 37)},
			{'7', cv::Rect(184, 63, 21, 37)},
			{'8', cv::Rect(211, 63, 22, 37)},
			{'9', cv::Rect(238, 62, 22, 37)},
	};
}