/**************************************************************************************************/

#pragma once

/**************************************************************************************************/

/**************************************************************************************************/
/*                                           Headers                                              */
/**************************************************************************************************/
#include"Project_Headers.hpp"


namespace pi {
	/*************************************************************************************************/
	/*                                     Global variables                                          */
	/*************************************************************************************************/

	/*Gaussian kernel of 3x3 dimensions*/
	extern const cv::Mat gauss3x3;

	/*Gaussian kernel of 5x5 dimensions*/
	extern const cv::Mat gauss5x5;

	/*Sobel kernel for horizontal changes*/
	extern const cv::Mat Fx3x3;

	/*Sobel kernel for vertical changes*/
	extern const cv::Mat Fy3x3;

	/*the gradient of an image that consistd of magnitude and orientation of iamge*/
	struct gradient{
		cv::Mat orient;
		cv::Mat magnit;
	};
}