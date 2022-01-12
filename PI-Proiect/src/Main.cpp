#include "Helper.hpp"
#include "Constants.hpp"
#include "Gradient.hpp"
#include <map>


const int dimension = 7;

void apply_grayscale(cv::Mat& input, cv::Mat& output)
{
	cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
}

void apply_threshold(cv::Mat& input, cv::Mat& output)
{
	cv::threshold(input, output, 128, 255, cv::THRESH_OTSU);
}

void apply_filter(cv::Mat& input, cv::Mat& output)
{
	cv::filter2D(input, output, -1, pi::gauss3x3);
}

void apply_equalize(cv::Mat& input, cv::Mat& output)
{
	cv::equalizeHist(input, output);
}

void apply_canny(cv::Mat& input, cv::Mat& output)
{
	cv::Canny(input, output, 100, 210, 3);
}

struct FontData
{
	cv::Mat sample;

	std::unordered_map<char, cv::Mat> letters;
	std::unordered_map<char, pi::gradient> letter_gradients;
};

struct PlateData
{
	std::vector<cv::Mat> segmented_plates;
	cv::Mat plate_drawing;
};

struct LetterInfo
{
	cv::Mat unknown_letter;

	char letter = '?';
	double distance = 2500;

	char value_letter = '?';
	double value_distance = 2500;

	char mag_letter = '?';
	double mag_distance = 2500;

	char angle_letter = '?';
	double angle_distance = 2500;
};

struct PlateTextData
{
	std::vector<std::vector<LetterInfo>> plate_letters;
};

FontData initialize_font()
{
	FontData fontData;

	fontData.sample = cv::imread("Resources\\Mittelschrift_sample.png", cv::IMREAD_GRAYSCALE);

	for (auto& pair : pi::loadLetterRectangles("Resources\\Mittelschrift_regions.txt"))
	{
		fontData.letters[pair.first] = cv::Mat(fontData.sample, pair.second);
	}

	for (auto& pair : fontData.letters)
	{
		fontData.letter_gradients[pair.first] = pi::contour_gradient(pair.second, 0);
	}

	return fontData;
}

PlateData detect_plate(const FontData& fontData, const cv::Mat& sample)
{
	PlateData plateData;

	cv::Mat result;
	pi::OperationList process;

	process.AddStep(apply_grayscale);
	process.AddStep(apply_filter);
	process.AddStep(apply_equalize);
	process.AddStep(apply_canny);

	process.Run(sample, result);

	// Find contours using OpenCV

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(result, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Simplify contours - multiple straight (or almost straight) lines become a single line
	// Prune contours that are way too small

	pi::simplifyContours(contours);
	pi::pruneShort(contours, 60);

	// Draw resulting rectangles - these show the zones that can contain potential car plates

	cv::Mat drawing = sample.clone();
	cv::RNG rng(12345);

	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(0, 255, 0);

		if (!pi::isLikeARectangle(contours[i]))
		{
			color = cv::Scalar(0, 0, 255);
		}

		cv::drawContours(drawing, contours, (int)i, color, 2, cv::LINE_8, cv::noArray(), 0);

		uint64_t size = contours[i].size();
		for (int pindex = 0; pindex < size; pindex++)
		{
			cv::drawMarker(drawing, contours[i][pindex], cv::Scalar::all(255.0 * pindex / size), 2, 5, 1, 1);
		}
	}

	// Show results

	plateData.plate_drawing = drawing;

	// Cut the license plate out

	for (int i = 0; i < contours.size(); i++)
	{
		if (pi::isLikeARectangle(contours[i]))
		{
			auto rect = pi::getBoundingBox(contours[i]);

			cv::Mat plate = cv::Mat(sample, cv::Range(rect.y, rect.height + rect.y), cv::Range(rect.x, rect.width + rect.x));
			
			plateData.segmented_plates.push_back(plate);
		}
	}

	return plateData;
}

LetterInfo read_letter(const FontData& fontData, const cv::Mat& letter)
{
	LetterInfo letterInfo;
	letterInfo.unknown_letter = letter;

	for (auto& pair : fontData.letters)
	{
		auto temp = cv::Mat(letter);
		auto grad_info = pi::contour_gradient(temp, 0);

		double value_distance = powf(1.0f / 255.0f, 2.0f) * pi::getMappedDistance(pair.second, letter);

		double mag_distance = powf(1.0f / 255.0f, 2.0f) * pi::getMappedDistance(fontData.letter_gradients.at(pair.first).magnit, grad_info.magnit);
		double angle_distance = powf(1.0f / 360.0f, 2.0f) * pi::getMappedDistance(fontData.letter_gradients.at(pair.first).orient, grad_info.orient);

		double finalDistance = value_distance * 0.6 + mag_distance * 0.25 + angle_distance * 0.5;

		if (value_distance < letterInfo.value_distance)
		{
			letterInfo.value_distance = value_distance;
			letterInfo.value_letter = pair.first;
		}

		if (mag_distance < letterInfo.mag_distance)
		{
			letterInfo.mag_distance = mag_distance;
			letterInfo.mag_letter = pair.first;
		}

		if (angle_distance < letterInfo.angle_distance)
		{
			letterInfo.angle_distance = angle_distance;
			letterInfo.angle_letter = pair.first;
		}

		if (finalDistance < letterInfo.distance)
		{
			letterInfo.distance = finalDistance;
			letterInfo.letter = pair.first;
		}
	}

	return letterInfo;
}

PlateTextData detect_and_read_text(const FontData& fontData, const PlateData& plateData)
{
	PlateTextData plateTextData;

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	pi::OperationList wordProcess;

	wordProcess.AddStep(apply_grayscale);
	wordProcess.AddStep(apply_threshold);
	wordProcess.AddStep(apply_canny);

	for (auto& plate : plateData.segmented_plates)
	{
		plateTextData.plate_letters.push_back(std::vector<LetterInfo>());
		auto& letterList = *plateTextData.plate_letters.rbegin();

		cv::Mat result;
		wordProcess.Run(plate, result);

		cv::findContours(result, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		// Simplify contours - multiple straight (or almost straight) lines become a single line

		// pi::simplifyContours(contours); // - Can't do due to sensitive algorithm!
		pi::pruneShort(contours, 20);

		// Calculate median height and width
		// The range of heights for letters is small
		// However, letters can have varying widths

		std::vector<double> heights;
		std::vector<double> widths;

		for (auto& contour : contours)
		{
			auto bbox = pi::getBoundingBox(contour);

			heights.push_back(bbox.height);
			widths.push_back(bbox.width);
		}

		std::sort(heights.begin(), heights.end());
		std::sort(widths.begin(), widths.end());

		int median_index = widths.size() / 2;
		double median_height = heights[median_index];
		double median_width = widths[median_index];

		// std::cout << "Median Width: " << median_width << " | Median Height: " << median_height << std::endl;

		if (widths.size() >= 3)
		{
			// Use the average of the median and the left / right value

			median_width = (widths[median_index - 1LL] + median_width + widths[median_index + 1LL]) / 3.0;
		}


		double width_threshold = 0.75;
		double height_threshold = 0.1;

		for (int i = 0; i < contours.size(); i++)
		{
			auto bbox = pi::getBoundingBox(contours[i]);

			// The height for a font varies by a low amount
			// Meanwhile the width can vary by a high amount
			// Therefore, drop all contours that vary:
			// * by >10% for height, or
			// * by >75% for width
			// Those are unlikely to be letters

			double width_variance = abs(bbox.width - median_width) / median_width;
			double height_variance = abs(bbox.height - median_height) / median_height;

			bool not_a_letter =
				width_variance >= width_threshold ||
				height_variance > height_threshold;

			if (not_a_letter)
				continue;

			auto letter_rect = cv::Rect(bbox.x, bbox.y, bbox.width, bbox.height);

			cv::Mat unknown_letter = cv::Mat(plate, letter_rect);
			cv::cvtColor(unknown_letter, unknown_letter, cv::COLOR_BGR2GRAY);

			LetterInfo letterInfo = read_letter(fontData, unknown_letter);

			letterList.push_back(letterInfo);
		}
	}

	return plateTextData;
}



int main(int argc, char** argv) {
	cv::CommandLineParser parser(argc, argv, "{@fileinput || input image}");

	// Read image

	std::string file;

	cv::Mat sample;

	if (parser.has("@fileinput"))
	{
		file = parser.get<cv::String>(0);
	}
	else
	{
		while (true)
		{
			std::cout << "Source image: ";
			std::getline(std::cin, file);
			
			if ((sample = cv::imread(file)).empty())
			{
				std::cerr << "Failed to open " << file << "!";
			}
			else break;
		}
	}

	FontData fontData = initialize_font();

	// Actual processing

	try {

		// Step 1 : detect plate(s)

		PlateData plateData = detect_plate(fontData, sample);

		// Step 2 : read text from plate(s)

		PlateTextData plateTextData = detect_and_read_text(fontData, plateData);

		// Step 3 : output info!

		std::cout << "Plates detected: " << plateTextData.plate_letters.size() << std::endl;

		cv::imshow("Plate detection", plateData.plate_drawing);

		for (int i = 0; i < plateTextData.plate_letters.size(); i++)
		{
			std::cout << " === Plate " << i << " ===" << std::endl;

			cv::imshow(std::string("Plate ") + std::to_string(i), plateData.segmented_plates[i]);

			auto& letterList = plateTextData.plate_letters[i];

			for (int j = 0; j < letterList.size(); j++)
			{
				auto& letterInfo = letterList[j];

				std::cout << " => Letter " << j << std::endl;
				std::cout << "Letter: " << letterInfo.letter << std::endl;
				std::cout << "Distance: " << letterInfo.distance << std::endl;

				std::cout << "Value Letter: " << letterInfo.value_letter << std::endl;
				std::cout << "Value Distance: " << letterInfo.value_distance << std::endl;

				std::cout << "Mag Letter: " << letterInfo.mag_letter << std::endl;
				std::cout << "Mag Distance: " << letterInfo.mag_distance << std::endl;

				std::cout << "Angle Letter: " << letterInfo.angle_letter << std::endl;
				std::cout << "Angle Distance: " << letterInfo.angle_distance << std::endl;

				cv::Mat letter_display;
				cv::resize(letterInfo.unknown_letter, letter_display, cv::Size(), 6.0, 6.0, 0);

				cv::imshow(std::string("P") + std::to_string(i) + ":L"  + std::to_string(j) + " | " + letterInfo.letter, letter_display);
			}
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