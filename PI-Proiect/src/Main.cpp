#include "Helper.hpp"
#include "Constants.hpp"
#include "Gradient.hpp"


const int dimension = 7;

int main(int argc, char** argv) {

	cv::Mat font_sample = cv::imread("Resources\\Mittelschrift_sample.png", cv::IMREAD_GRAYSCALE);

	std::unordered_map<char, cv::Mat> letter_regions;
	std::unordered_map<char, pi::gradient> letter_regions_grad;

	for (auto& pair : pi::loadLetterRectangles("Resources\\Mittelschrift_regions.txt")) {
		letter_regions[pair.first] = cv::Mat(font_sample, pair.second);
	}

	for (auto& pair : letter_regions_grad) {
		cv::imshow("magnitude " + std::to_string(rand() % 1000), pair.second.magnit);
		cv::imshow("orientation " + std::to_string(rand() % 1000), pair.second.orient);
	}

	auto grayscaleStep = [](cv::Mat& input, cv::Mat& output)
	{
		cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
	};

	auto thresholdStep = [](cv::Mat& input, cv::Mat& output)
	{
		cv::threshold(input, output, 128, 255, cv::THRESH_OTSU);
	};

	auto filterStep = [&](cv::Mat& input, cv::Mat& output)
	{
		cv::filter2D(input, output, -1, pi::gauss3x3);
	};

	auto equalizeStep = [](cv::Mat& input, cv::Mat& output)
	{
		cv::equalizeHist(input, output);
	};

	auto cannyStep = [](cv::Mat& input, cv::Mat& output)
	{
		cv::Canny(input, output, 100, 210, 3);
	};

	auto sharpenStep = [&](cv::Mat& input, cv::Mat& output)
	{
		cv::filter2D(input, output, -1, pi::sharpen3x3);
		cv::add(input, output, output);
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

		// Calculate median height and width
		// The range of heights for letters is small
		// However, letters can have varying widths

		std::vector<double> heights;
		std::vector<double> widths;

		for (auto& contour : contours) {
			heights.push_back(pi::getBoundingBox(contour).height);
			widths.push_back(pi::getBoundingBox(contour).width);
		}

		std::sort(heights.begin(), heights.end());
		std::sort(widths.begin(), widths.end());

		int median_index = widths.size() / 2;

		double median_height = heights[median_index];
		double median_width = widths[median_index];

		std::cout << "Median Width: " << median_width << " | Median Height: " << median_height << std::endl;

		if (widths.size() > 3) {
			// Combine the median with the average of the element to the left and the right
			// This provides a slightly better value for a letter's width

			median_width = (widths[median_index - 1LL] + median_width + widths[median_index + 1LL]) / 3.0;
		}

		// Drop all contours that vary by >10% for height, or >75% for width
		// Those are unlikely to be letters

		for (int i = 0; i < contours.size(); i++) {
			auto box = pi::getBoundingBox(contours[i]);

			if (abs(box.width - median_width) / median_width >= 0.75 || abs(box.height - median_height) / median_height > 0.1) {
				contours.erase(contours.begin() + i);
				i--;
				std::cout << "Skipped a bounding box: " << box.width << ", " << box.height << " at " << box.x << ", " << box.y << std::endl;
			}
		}

		std::cout << median_width << " - " << median_height << std::endl;

		for (auto& contour : contours) {
			auto rect = pi::getBoundingBox(contour);

			auto lol = std::to_string(rand() % 1000);

			std::cout << rect.width << " " << rect.height << std::endl;

			double width_factor = abs(rect.width - median_width) / median_width;
			double height_factor = abs(rect.height - median_height) / median_height;

			// The height for a font varies by a low amount
			// Meanwhile the width can vary by a high amount

			cv::Mat image = cv::Mat(plate, cv::Range(rect.y, rect.height + rect.y), cv::Range(rect.x, rect.width + rect.x));

			cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

			cv::Mat region = pi::getRegionFeatures(image, dimension);

			std::cout << std::setprecision(2) << std::fixed;

			double lowest_distance = 2500.0;
			char selected_character = '?';


			for (auto& pair : letter_regions) {
				cv::Mat oldRegion = pi::getRegionFeatures(pair.second, dimension);
				//double distance = pi::getLetterDistance(oldRegion, region);
				double newDistance = pi::getMappedDistance(pair.second, image);

				if (newDistance < lowest_distance) {
					lowest_distance = newDistance;
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