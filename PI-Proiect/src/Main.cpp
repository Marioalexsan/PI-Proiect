#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iomanip>
#include "Helper.hpp"


int main(int argc, char** argv) {

	// Utility Stuff

	cv::Mat gauss3x3 = cv::Mat_<double>(
		{
			1, 2, 1,
			2, 4, 2,
			1, 2, 1
		}
	).reshape(0, 3) * (1.0 / 16.0);

	cv::Mat gauss5x5 = cv::Mat_<double>(
		{
			2, 4, 5, 4, 2,
			4, 9, 12, 9, 4,
			5, 12, 15, 12, 5,
			4, 9, 12, 9, 4,
			2, 4, 5, 4, 2
		}
	).reshape(0, 5) * (1.0 / 159.0);

	cv::Mat average3x3 = cv::Mat_<double>(
		{
			1, 1, 1,
			1, 1, 1,
			1, 1, 1
		}
	).reshape(0, 3) * (1.0 / 9.0);

	cv::Mat sharpen3x3 = cv::Mat_<double>(
		{
			-1, -1, -1,
			-1, 9, -1,
			-1, -1, -1
		}
	).reshape(0, 3) * (1.0 / 9.0);



	auto grayscaleStep = [](cv::Mat& input, cv::Mat& output)
	{
		cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
		cv::imshow("Grayscale", output);
	};

	auto thresholdStep = [](cv::Mat& input, cv::Mat& output)
	{
		cv::threshold(input, output, 128, 255, cv::THRESH_OTSU);
		cv::imshow("Threshold", output);
	};

	auto filterStep = [&](cv::Mat& input, cv::Mat& output)
	{
		cv::filter2D(input, output, -1, gauss3x3);
		cv::imshow("Filter", output);
	};

	auto equalizeStep = [](cv::Mat& input, cv::Mat& output)
	{
		cv::equalizeHist(input, output);
		cv::imshow("Equalize", output);
	};

	auto cannyStep = [](cv::Mat& input, cv::Mat& output)
	{
		cv::Canny(input, output, 100, 210, 3);
		cv::imshow("Canny", output);
	};

	auto sharpenStep = [&](cv::Mat& input, cv::Mat& output)
	{
		cv::filter2D(input, output, -1, sharpen3x3);
		cv::add(input, output, output);
		cv::imshow("Sharpen", output);
	};

	// Actual processing

	try {

		// Read image

		std::string file;
		cv::CommandLineParser parser(argc, argv, "{@fileinput || input image}");

		if (parser.has("@fileinput")) {
			file = parser.get<cv::String>(0);
		}
		else {
			std::cout << "Source image: ";
			std::getline(std::cin, file);
		}

		cv::Mat original = cv::imread(file);

		if (original.empty()) {
			std::cerr << "Failed to open " << file << "!";
			cv::waitKey();
			return 0;
		}

		// Do processing

		pi::ImageProcess process;

		process.AddStep(grayscaleStep);
		process.AddStep(filterStep);
		process.AddStep(equalizeStep);
		process.AddStep(cannyStep);

		cv::Mat result;
		process.Run(original, result);

		// Find contours using OpenCV

		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(result, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		bool compress = true;
		bool pruneShort = true;

		// Simplify contours - multiple straight (or almost straight) lines become a single line

		if (compress) {
			pi::simplifyContours(contours);
		}

		pi::pruneEmpty(contours);

		// Prune contours that are way too small

		if (pruneShort) {
			pi::pruneShort(contours, 60);
		}

		// Draw resulting rectangles - these show the zones that can contain potential car plates

		cv::Mat drawing = original.clone();
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

		cv::Mat plate;

		for (int i = 0; i < contours.size(); i++) {
			if (pi::isLikeARectangle(contours[i])) {
				auto rect = pi::getBoundingBox(contours[i]);

				plate = cv::Mat(original, cv::Range(rect.y, rect.height + rect.y), cv::Range(rect.x, rect.width + rect.x));

				cv::imshow("License Plate", plate);

				std::cout << "Color distance from white: " << pi::getColorMatch(plate, cv::Scalar::all(255));

				break;
			}
		}

		// Step 2 : read text

		if (plate.empty()) {
			cv::waitKey();
			return 0;
		}

		pi::ImageProcess wordProcess;

		wordProcess.AddStep(grayscaleStep);
		wordProcess.AddStep(thresholdStep);
		wordProcess.AddStep(cannyStep);

		cv::Mat wordResult;
		wordProcess.Run(plate, wordResult);

		cv::findContours(wordResult, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		compress = true;
		pruneShort = true;

		// Simplify contours - multiple straight (or almost straight) lines become a single line

		if (compress) {
			pi::simplifyContours(contours);
		}

		pi::pruneEmpty(contours);

		// Prune contours that are way too small

		if (pruneShort) {
			pi::pruneShort(contours, 20);
		}

		std::vector<pi::ImageLetter> letters;

		for (auto& contour : contours) {
			pi::ImageLetter il;

			auto rect = pi::getBoundingBox(contour);
			auto letter = cv::Mat(plate, cv::Range(rect.y, rect.height + rect.y), cv::Range(rect.x, rect.width + rect.x));


			cv::cvtColor(letter, letter, cv::COLOR_BGR2GRAY);
			cv::threshold(letter, letter, 0.0, 255.0, cv::THRESH_OTSU);
			//pi::thinningAlgorithm(letter, letter);

			il.image = letter;

			pi::computeRegions(il, 7, 7);

			auto lol = std::to_string(rand() % 1000);

			std::cout << std::setprecision(2) << std::fixed;

			std::cout << "Computed regions for letter " << lol << std::endl;

			for (int y = 0; y < il.regionRows; y++) {
				for (int x = 0; x < il.regionCols; x++) {
					std::cout << il.regions[x + il.regionCols * y] << " ";
				}
				std::cout << std::endl;
			}
			
			std::cout << std::endl;

			cv::resize(letter, letter, cv::Size(), 4.f, 4.f, 1);

			cv::imshow(lol + " Letter", letter);

			il.image = letter;
			letters.push_back(il);
		}

		cv::imshow("Word result", wordResult);
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