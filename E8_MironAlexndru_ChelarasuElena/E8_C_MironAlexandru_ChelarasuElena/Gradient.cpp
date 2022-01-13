/**************************************************************************************************/
/*                                           Headers                                              */
/**************************************************************************************************/
#include"Gradient.hpp"

/*************************************************************************************************/
/*                                       Defines & types                                         */
/*************************************************************************************************/

// Defines suck. We're in C++ land dangit -Mario
#define BASE_VAL 0
#define ALPHA 0.5
#define BETA 0.5
#define GAMA 0
#define PRECISION 1
#define WIDTH 6


namespace pi {

	cv::Mat calculate_magnitude(cv::Mat Sx, cv::Mat Sy)
	{
		cv::Mat magnitude = cv::Mat(Sx.rows, Sx.cols, CV_64F);

		for (int y = BASE_VAL; y < Sx.rows; ++y)
		{
			for (int x = BASE_VAL; x < Sx.cols; ++x)
			{
				double valX_Sx = (double)Sx.at<uchar>(y, x);
				double valY_Sy = (double)Sy.at<uchar>(y, x);
				
				magnitude.at<double>(y, x) = sqrt(valX_Sx * valX_Sx + valY_Sy * valY_Sy);
			}
		}

		return magnitude;
	}
	cv::Mat calculate_orientation(cv::Mat Sx, cv::Mat Sy) 
	{
		cv::Mat orientation = cv::Mat(Sx.rows, Sx.cols, CV_64F);
		for (int y = BASE_VAL; y < Sx.rows; ++y)
		{
			for (int x = BASE_VAL; x < Sx.cols; ++x)
			{
				double valX = (double)Sx.at<uchar>(y, x);
				double valY = (double)Sy.at<uchar>(y, x);

				//calculez unghiul theta
				orientation.at<double>(y, x) = (double) cv::fastAtan2(valY, valX);
			}
		}

		return orientation;
	}

	pi::gradient contour_gradient(cv::Mat& image)
	{
		cv::Mat out; 
		cv::Mat Sx;
		cv::Mat Sy;

		//calculam derivatele in directiile x si y
		cv::filter2D(image, Sx, -1, pi::Fx3x3);
		cv::filter2D(image, Sy, -1, pi::Fy3x3);

		cv::convertScaleAbs(Sx, Sx);
		cv::convertScaleAbs(Sy, Sy);

		cv::addWeighted(Sx, ALPHA, Sy, BETA, GAMA, out);

		//calculez magnitudinea si orientatia (theta)

		cv::Mat orientation = calculate_orientation(Sx, Sy);
		cv::Mat magnitude = calculate_magnitude(Sx, Sy);

		pi::gradient output;

		output.magnit = magnitude;
		output.orient = orientation;

		return output;
	}

	/*function that imports the data from a gradient type to a file*/
	void toFile(pi::gradient findings)
	{
		std::ofstream file("letters.txt", std::ofstream::app);

		file.precision(PRECISION);
		file.width(WIDTH);
		file.setf(std::ofstream::fixed);

		if (!file.good())
		{
			std::cout << "\nUnable to open file letters.txt";
			return;
		}

		file << "Magnitudine: " << std::endl;

		for (int index = BASE_VAL; index < findings.magnit.rows; ++index)
		{
			for (int jindex = BASE_VAL; jindex < findings.magnit.cols; ++jindex)
			{
				file << findings.magnit.at<double>(index, jindex) << ' ';
			}

			file << std::endl;
		}

		file << "Orientation: " << std::endl;

		for (int index = BASE_VAL; index < findings.orient.rows; ++index)
		{
			for (int jindex = BASE_VAL; jindex < findings.orient.cols; ++jindex)
			{
				file << findings.orient.at<double>(index, jindex) << ' ';
			}

			file << std::endl;
		}

		file.close();
	}
}

