#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>

int main(int argc, char** argv) {
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

	// Debug output

	std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
	std::clog << "Stage 2 Kernel is " << stage2Kernel3x3 << std::endl;

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

	// Apply filter
	cv::filter2D(stage1, stage2, -1, stage2Kernel5x5);
	//stage1.copyTo(stage2);

	// Show results

	cv::imshow("Sample", img);
	cv::imshow("Grayscale", stage1);
	cv::imshow("Noise reduction", stage2);

	// Contour code, taken from opencv docs

	cv::Mat canny_output;
	cv::Canny(stage2, canny_output, 100, 200);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);

	cv::RNG rng(12345);
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		cv::drawContours(drawing, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0);
	}

	cv::imshow("Contours", drawing);

	cv::waitKey();

	return 0;
}