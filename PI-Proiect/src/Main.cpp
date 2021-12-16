#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iomanip>
#include "Helper.hpp"
#include "Constants.hpp"
#include <unordered_map>


int main(int argc, char** argv) {

	cv::Mat font_sample = cv::imread("Resources\\Mittelschrift_sample.png", cv::IMREAD_GRAYSCALE);

	std::unordered_map<char, cv::Mat> letter_regions;

	for (auto& pair : pi::letter_sheet) {
		cv::Mat letter_image = cv::Mat(font_sample, pair.second);

		letter_regions[pair.first] = pi::getRegionFeatures(letter_image, 7, 7);
	}

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
		cv::filter2D(input, output, -1, pi::gauss3x3);
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
		cv::filter2D(input, output, -1, pi::sharpen3x3);
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

		pi::OperationList process;

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

				std::cout << "Color distance from white: " << pi::getColorMatch(plate, cv::Scalar::all(255)) << std::endl;

				break;
			}
		}

		// Step 2 : read text

		if (plate.empty()) {
			cv::waitKey();
			return 0;
		}

		pi::OperationList wordProcess;

		wordProcess.AddStep(grayscaleStep);
		wordProcess.AddStep(thresholdStep);
		wordProcess.AddStep(cannyStep);

		cv::Mat wordResult;
		wordProcess.Run(plate, wordResult);

		cv::findContours(wordResult, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		compress = false;
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

		std::vector<cv::Mat> letters;

		for (auto& contour : contours) {
			auto rect = pi::getBoundingBox(contour);

			cv::Mat image = cv::Mat(plate, cv::Range(rect.y, rect.height + rect.y), cv::Range(rect.x, rect.width + rect.x));

			cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
			cv::threshold(image, image, 0.0, 255.0, cv::THRESH_OTSU);
			//pi::thinningAlgorithm(letter, letter);

			cv::Mat region = pi::getRegionFeatures(image, 7, 7);

			auto lol = std::to_string(rand() % 1000);

			std::cout << std::setprecision(2) << std::fixed;

			double lowest_distance = 2500.0;
			char selected_character = '?';

			for (auto& pair : letter_regions) {
				double distance = pi::getLetterDistance(pair.second, region);

				if (distance < lowest_distance) {
					lowest_distance = distance;
					selected_character = pair.first;
				}
			}

			cv::resize(image, image, cv::Size(), 4.f, 4.f, 0);
			cv::imshow("Letter" + lol, image);

			std::cout << "Letter " << lol << " is " << selected_character << ", distance: " << lowest_distance << std::endl;

			letters.push_back(region);
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