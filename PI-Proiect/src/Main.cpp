#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include "Helper.hpp"


int main(int argc, char** argv) {
	try {
		cv::CommandLineParser parser(argc, argv, "{@fileinput || input image}");

		cv::Mat img; // Imaginea originala
		cv::Mat stage1; // Grayscale
		cv::Mat stage2; // Noise reduction

		cv::Mat stage2Kernel3x3 = cv::Mat_<double>(
			{
				1, 2, 1,
				2, 4, 2,
				1, 2, 1
			}
		).reshape(0, 3) * (1.0 / 16.0);

		cv::Mat stage2Kernel5x5 = cv::Mat_<double>(
			{
				2, 4, 5, 4, 2,
				4, 9, 12, 9, 4,
				5, 12, 15, 12, 5,
				4, 9, 12, 9, 4,
				2, 4, 5, 4, 2
			}
		).reshape(0, 5) * (1.0 / 159.0);

		cv::Mat kernel3x3smoother = cv::Mat_<double>(
			{
				1, 1, 1,
				1, 1, 1,
				1, 1, 1
			}
		).reshape(0, 3) * (1.0 / 9.0);


		// Read image

		std::string file;

		if (parser.has("@fileinput")) {
			file = parser.get<cv::String>(0);
		}
		else {
			std::cout << "Source image: ";
			std::getline(std::cin, file);
		}

		img = cv::imread(file);

		if (img.empty()) {
			std::cerr << "Failed to open " << file << "!";
			cv::waitKey();
			return 0;
		}

		// Do processing

		cv::cvtColor(img, stage1, cv::COLOR_BGR2GRAY);

		cv::Mat hsv_img;
		cv::cvtColor(img, hsv_img, cv::COLOR_BGR2HSV);

		// Apply filter

		cv::filter2D(stage1, stage2, -1, stage2Kernel3x3);

		// Increase contrast

		pi::applyContrast(stage2, 80, 160, 50, 220);

		cv::imshow("Contrast", stage2);

		// Apply Canny algorithm to find edges

		cv::Mat canny_output;
		cv::Canny(stage2, canny_output, 100, 210, 3);

		cv::imshow("Canny", canny_output);

		// Find contours using OpenCV

		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(canny_output, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		bool compress = true;
		bool pruneShort = true;

		// Compress contours to straight lines

		if (compress) {
			pi::compressContours(contours);
		}

		// Prune contours that are too small

		if (pruneShort) {
			pi::pruneShort(contours, 200);
		}

		// Draw resulting rectangles - these show the zones that can contain potential car plates

		cv::Mat drawing = img.clone();
		cv::RNG rng(12345);

		for (size_t i = 0; i < contours.size(); i++)
		{
			cv::Scalar color = cv::Scalar(0, 255, 0);

			if (!pi::isLikeARectangle(contours[i])) {
				color = cv::Scalar(0, 0, 255);
			}

			cv::drawContours(drawing, contours, (int)i, color, 2, cv::LINE_8, cv::noArray(), 0);

			uint64_t size = contours[i].size();
			for (int pindex = 0; pindex < size; pindex++) {
				cv::drawMarker(drawing, contours[i][pindex], cv::Scalar::all(255.0 * pindex / size), 2, 5, 1, 1);
			}
		}
		
		// Show results

		cv::imshow("Contours", drawing);

		// Cut the license plate out

		if (contours.size() > 0) {
			auto rect = pi::getBoundingBox(contours[0]);

			cv::Mat cut = cv::Mat(img, cv::Range(rect.y, rect.height + rect.y), cv::Range(rect.x, rect.width + rect.x));

			cv::imshow("License Plate", drawing);
		}


	}
	catch (cv::Exception& e) {
		std::cerr << "An OpenCV exception occurred! " << e.what();
	}
	catch (std::exception& e) {
		std::cerr << "An exception occurred! " << e.what();
	}

	cv::waitKey();

	return 0;
}