#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>

double cosine(cv::Point a, cv::Point b, cv::Point c) {
	cv::Point vec_ab = a - b;
	cv::Point vec_bc = c - b;

	return vec_ab.ddot(vec_bc) / sqrt(vec_ab.ddot(vec_ab) * vec_bc.ddot(vec_bc));
}

double perimeter(std::vector<cv::Point>& points) {
	double perimeter = 0;

	for (int i = 0; i < points.size(); i++) {
		cv::Point dist = points[(i + 1) % points.size()] - points[i];
		perimeter += sqrt((dist).ddot(dist));
	}
	
	return perimeter;
}

bool probably_a_rectangle(std::vector<cv::Point>& points) {
	// Rectangle criteria:
	// - Opposing edges must be almost parallel
	// - Adjacent edges must be aproximately 90 degrees apart

	if (points.size() != 4) {
		return false;
	}

	cv::Point a = points[0];
	cv::Point b = points[1];
	cv::Point c = points[2];
	cv::Point d = points[3];

	const double thresh = 0.18;

	bool bad_angles =
		abs(cosine(a, b, c)) > thresh ||
		abs(cosine(b, c, d)) > thresh ||
		abs(cosine(c, d, a)) > thresh ||
		abs(cosine(d, a, b)) > thresh;

	return !bad_angles;
}

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
	cv::findContours(canny_output, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);

	// Reduce contours

	for (int cindex = 0; cindex < contours.size(); cindex++) {
		std::vector<cv::Point>& contour = contours[cindex];

		for (int pindex = 0; pindex < contour.size(); pindex++) {
			cv::Point a = contour[pindex];
			cv::Point b = contour[(pindex + 1) % contour.size()];
			cv::Point c = contour[(pindex + 2) % contour.size()];

			cv::Vec2d vec_ab = cv::Vec2d(a.x - b.x, a.y - b.y);
			cv::Vec2d vec_bc = cv::Vec2d(b.x - c.x, b.y - c.y);

			double ab_len = sqrt(vec_ab.ddot(vec_ab));
			double bc_len = sqrt(vec_bc.ddot(vec_bc));

			double cosine = vec_ab.ddot(vec_bc) / (ab_len * bc_len);

			if (ab_len < 3 || bc_len < 3 || abs(cosine) >= 0.650) {
				contour.erase(contour.begin() + (pindex + 1) % contour.size());
				pindex -= 1;
			}
		}
	}

	// Prune contours that have an edge count other than 4

	for (int cindex = 0; cindex < contours.size(); cindex++) {
		if (contours[cindex].size() != 4) {
			contours.erase(contours.begin() + cindex);
			cindex -= 1;
		}
	}

	// Prune contours that are too small

	for (int cindex = 0; cindex < contours.size(); cindex++) {
		if (perimeter(contours[cindex]) < 200) {
			contours.erase(contours.begin() + cindex);
			cindex -= 1;
		}
	}

	// Prune contours that don't approximate a rectangle
	
	for (int cindex = 0; cindex < contours.size(); cindex++) {
		if (!probably_a_rectangle(contours[cindex])) {
			contours.erase(contours.begin() + cindex);
			cindex -= 1;
		}
	}

	// Show contours

	cv::RNG rng(12345);
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Scalar color;

		if (contours[i].size() == 4) {
			color = cv::Scalar(0, 255, 0);
		}
		else if (contours[i].size() == 5) {
			color = cv::Scalar(255, 0, 0);
		}
		else if (contours[i].size() == 6) {
			color = cv::Scalar(255, 255, 0);
		}
		else {
			color = cv::Scalar(0, 0, 255);
		}

		cv::drawContours(drawing, contours, (int)i, color, 2, cv::LINE_8, cv::noArray(), 0);

		int size = contours[i].size();
		for (int pindex = 0; pindex < size; pindex++) {
			double value = 128 * pindex / (double)size;
			cv::drawMarker(drawing, contours[i][pindex], cv::Scalar(127 + value, 127 + value, 127 + value), 2, 5, 1, 1);
		}
	}

	cv::imshow("Contours", drawing);

	cv::waitKey();

	return 0;
}