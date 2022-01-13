#include"Project_Headers.hpp"
#include "Constants.hpp"


namespace pi {
	/*************************************************************************************************/
	/*                                     Global variables                                          */
	/*************************************************************************************************/

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