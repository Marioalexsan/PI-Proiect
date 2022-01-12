#include"Gradient.hpp"

namespace pi {

	cv::Mat calculate_magnitude(cv::Mat Sx, cv::Mat Sy)
	{
		cv::Mat magnitude = cv::Mat(Sx.rows, Sx.cols, CV_64F);
		cv::pow(Sx, 2.0, Sx);
		cv::pow(Sy, 2.0, Sy);


		for (int y = 0; y < Sx.rows; ++y)
		{
			for (int x = 0; x < Sx.cols; ++x)
			{
				double valX_Sx = (double)Sx.at<uchar>(y, x);
				double valY_Sy = (double)Sy.at<uchar>(y, x);
				
				magnitude.at<double>(y, x) = sqrt(valX_Sx + valY_Sy);
				magnitude.at<double>(y, x) = sqrt(valX_Sx + valY_Sy);
			}
		}

		return magnitude;
	}
	cv::Mat calculate_orientation(cv::Mat Sx, cv::Mat Sy) 
	{
		cv::Mat orientation = cv::Mat(Sx.rows, Sx.cols, CV_64F);
		for (int y = 0; y < Sx.rows; ++y)
		{
			for (int x = 0; x < Sx.cols; ++x)
			{
				double valX = (double)Sx.at<uchar>(y, x);
				double valY = (double)Sy.at<uchar>(y, x);
				//calculez unghiul theta
				orientation.at<double>(y, x) = (double) cv::fastAtan2(valX, valY) * (180/CV_PI);
			}
		}

		return orientation;
	}

	pi::gradient contour_gradient(cv::Mat& image, int dimension)
	{
		cv::Mat out; //ERR: daca rows si cols nu sunt egale
		//cv::Mat region = cv::Mat::zeros(dimension, dimension, CV_64F);
		cv::Mat Sx;
		cv::Mat Sy;

		//calculam derivatele in directiile x si y
		cv::filter2D(image, Sx, -1, pi::Fx3x3);
		cv::filter2D(image, Sy, -1, pi::Fy3x3);

		cv::convertScaleAbs(Sx, Sx);
		cv::convertScaleAbs(Sy, Sy);

		cv::addWeighted(Sx, 0.5, Sy, 0.5, 0, out);

		//calculez magnitudinea si orientatia (theta)

		cv::Mat orientation = calculate_orientation(Sx, Sy);
		cv::Mat magnitude = calculate_magnitude(Sx, Sy);
		
		cv::resize(magnitude, magnitude, cv::Size(), 8.0, 8.0, 0);
		cv::resize(orientation, orientation, cv::Size(), 8.0, 8.0, 0);


		//cv::imshow("magnitude" + std::to_string(rand()%1000), magnitude);
		//cv::imshow("orientation" + std::to_string(rand() % 1000), orientation);
		pi::gradient output;

		output.magnit = magnitude;
		output.orient = orientation;

		return output;
	}

	/*function that imports the data prom a gradient type to a file*/
	void toFile(pi::gradient findings)
	{
		std::ofstream file("letters.txt");
		//file.open("letters.txt");
		if (file.is_open()) 
		{
			file << "\nMagnitudine: ";
			for (int index = 0; index <= findings.magnit.rows; ++index)
			{
				file << "\n";
				for (int jindex = 0; jindex < findings.magnit.cols; ++jindex)
				{
					file << findings.magnit.at<double>(index, jindex)<<" ";
				}
			}

			file << "\nOrientation: ";
			for (int index = 0; index < findings.orient.rows; ++index)
			{
				file << "\n";
				for (int jindex = 0; jindex <= findings.magnit.cols; ++jindex)
				{
					file << findings.orient.at<double>(index, jindex) << " ";
				}
			}

			file.close();
		}
		else
		{
			std::cout << "\nUnable to open file letters.txt";
		}
	}
}

