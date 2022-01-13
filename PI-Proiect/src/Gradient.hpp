/**************************************************************************************************/

#pragma once

/**************************************************************************************************/
/*                                           Headers                                              */
/**************************************************************************************************/
#include "Project_Headers.hpp"
#include "Constants.hpp"

/**************************************************************************************************/
/*                                 Private Functions Declaration                                  */
/**************************************************************************************************/

namespace pi {
	 /**
	 * \brief Function that writes to a file a gradient type object
	 *
	 * \param[in] findings - received data containing two Mat's
	 *
	 * \note This function prints separatelly the magnitude Mat and orientation Mat of a region
	 */
	void toFile(pi::gradient findings);

	/**
	 * \brief Function that calculates the magnitude of an image
	 *
	 * \param[in] Sx - derivate in the x direction
	 *
	 * \param[in] Sy - derivate in the x direction
	 *
	 * \param[out] magnitude - magnitude of the image
	 *
	 * \note This function calculates the root of the ((pixel)Sx^2 + (pixel)Sy^2)
	 */
	cv::Mat calculate_magnitude(cv::Mat Sx, cv::Mat Sy);

	/**
	 * \brief Function that calculates the orientation of an image
	 *
	 * \param[in] Sx - derivate in the x direction
	 * 
	 * \param[in] Sy - derivate in the x direction
	 * 
	 * \param[out] orientation - orientation of the image
	 *
	 * \note This function calculates the arctangent of every pixel of Sx and Sy
	 */
	cv::Mat calculate_orientation(cv::Mat Sx, cv::Mat Sy);

	/**
	 * \brief Function that calculates the gradient of an image
	 *
	 * \param[in] image - received image/ region of the image
	 *
	 * \note First we do some helpings, before the calculus
	 */
	pi::gradient contour_gradient(cv::Mat& image);
}