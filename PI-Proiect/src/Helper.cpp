#include "Helper.hpp"

namespace pi {

	OperationList::OperationList() {}

	OperationList::~OperationList() {}

	void OperationList::Run(const cv::Mat& input, cv::Mat& output)
	{
		cv::Mat current = input.clone();
		cv::Mat next;

		for (auto& func : steps) {
			func(current, next);
			current = next;
		}

		output = current;
	}

	void OperationList::AddStep(std::function<void(cv::Mat&, cv::Mat&)> step) {
		steps.push_back(step);
	}

	void OperationList::Clear() {
		steps.clear();
	}

	/**************************************************************************************************/
	/*                                     Public Functions                                          */
	/**************************************************************************************************/

	std::unordered_map<char, cv::Rect> loadLetterRectangles(std::string path) {
		std::ifstream file(path);

		if (!file.good()) {
			std::cout << "Failed to open file " << path << std::endl;
			return std::unordered_map<char, cv::Rect>();
		}

		auto result = std::unordered_map<char, cv::Rect>();

		char ch;
		int x, y, width, height;

		try {
			while (!file.eof()) {
				file >> ch >> x >> y >> width >> height;
				result[ch] = cv::Rect(x, y, width, height);
			}
		}
		catch (...) {
			std::cout << "Failed to read file " << path << std::endl;
			return std::unordered_map<char, cv::Rect>();
		}

		return result;
	}

	double lineCos(cv::Point a, cv::Point b, cv::Point c) {
		cv::Point vec_ab = a - b;
		cv::Point vec_bc = c - b;

		return vec_ab.ddot(vec_bc) / sqrt(vec_ab.ddot(vec_ab) * vec_bc.ddot(vec_bc));
	}

	double contourPerimeter(const std::vector<cv::Point>& points) {
		double perimeter = 0;

		for (uint64_t i = 0; i < points.size(); i++) {
			cv::Point dist = points[(i + 1) % points.size()] - points[i];
			perimeter += sqrt((dist).ddot(dist));
		}

		return perimeter;
	}

	bool isLikeARectangle(const std::vector<cv::Point>& points) {
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

		const double thresh = 0.12;

		bool bad_angles =
			abs(lineCos(a, b, c)) > thresh ||
			abs(lineCos(b, c, d)) > thresh ||
			abs(lineCos(c, d, a)) > thresh ||
			abs(lineCos(d, a, b)) > thresh;

		return !bad_angles;
	}

	double getColorMatch(cv::Mat& img, cv::Scalar color) {
		int baseb = (int)color[0];
		int baseg = (int)color[1];
		int baser = (int)color[2];

		int baseluminosity = (baseb + baseg + baser) / 3;

		if (img.type() != CV_8UC3) {
			throw std::exception("Oh Shiet");
		}

		int matchingCount = 0;
		int total = 0;
		uint8_t* ptr = img.data;
		for (int j = 0; j < img.rows; j++) {
			for (int i = 0; i < img.cols; i++) {
				int b = ptr[(img.cols * j + i) * img.channels()];
				int g = ptr[(img.cols * j + i) * img.channels() + 1];
				int r = ptr[(img.cols * j + i) * img.channels() + 2];

				int luminosity = (b + g + r) / 3;


				int distance = abs(baseluminosity - luminosity);
				int threshold = 104;

				if (distance <= threshold) {
					matchingCount++;
				}
				total++;
			}
		}

		return matchingCount / (double)total;
	}

	void pruneNonRectangles(std::vector<std::vector<cv::Point>>& target) {

		// Prune contours that have an edge count other than 4

		for (int cindex = 0; cindex < target.size(); cindex++) {
			if (target[cindex].size() != 4) {
				target.erase(target.begin() + cindex);
				cindex -= 1;
			}
		}

		// Prune shapes that don't approximate a rectangle

		for (int cindex = 0; cindex < target.size(); cindex++) {
			if (!isLikeARectangle(target[cindex])) {
				target.erase(target.begin() + cindex);
				cindex -= 1;
			}
		}
	}

	void pruneEmpty(std::vector<std::vector<cv::Point>>& target) {
		for (int cindex = 0; cindex < target.size(); cindex++) {
			if (target[cindex].size() < 2) {
				target.erase(target.begin() + cindex);
				cindex -= 1;
			}
		}
	}

	void pruneShort(std::vector<std::vector<cv::Point>>& target, double threshold) {
		for (int cindex = 0; cindex < target.size(); cindex++) {
			if (pi::contourPerimeter(target[cindex]) < threshold) {
				target.erase(target.begin() + cindex);
				cindex -= 1;
			}
		}
	}

	void simplifyContours(std::vector<std::vector<cv::Point>>& target) {
		int passes = 16;
		double start_threshold = 0.98;
		double relax_per_pass = 0.26 / passes;

		for (int pass = 0; pass < passes; pass++) {
			for (int cindex = 0; cindex < target.size(); cindex++) {

				if (target[cindex].size() <= 2) {
					continue;
				}

				for (uint64_t pindex = 0; pindex < target[cindex].size(); pindex++) {
					std::vector<cv::Point>& contour = target[cindex];

					cv::Point a = contour[pindex];
					cv::Point b = contour[(pindex + 1) % contour.size()];
					cv::Point c = contour[(pindex + 2) % contour.size()];

					cv::Point ba = a - b;
					cv::Point bc = c - b;

					double current_thresh = start_threshold - relax_per_pass * pass;

					if (ba.ddot(ba) <= 9.0 || bc.ddot(bc) <= 9.0 || abs(lineCos(a, b, c)) >= current_thresh) {
						contour.erase(contour.begin() + (pindex + 1) % contour.size());
						pindex -= 1;
					}
				}
			}
		}
	}

	void applyContrast(cv::Mat& img, float a, float b, float sa, float sb) {
		if (img.type() != CV_8UC1) {
			return;
		}

		float m = sa / (float)a;
		float n = (sb - sa) / (float)a;
		float p = (255 - sb) / (255 - b);

		for (int y = 0; y < img.cols; y++) {
			for (int x = 0; x < img.rows; x++) {

				uint8_t r = img.at<uint8_t>(x, y);

				if (r <= a) {
					r = (uint8_t)std::min((int)(m * r), 255);
				}
				else if (r <= b) {
					r = (uint8_t)std::min((int)(n * (r - a) + sa), 255);
				}
				else {
					r = (uint8_t)std::min((int)(p * (r - b) + sb), 255);
				}

				img.at<uint8_t>(x, y) = r;
			}
		}
	}

	cv::Rect getBoundingBox(std::vector<cv::Point>& points) {
		int x_min = points[0].x;
		int x_max = x_min;
		int y_min = points[0].y;
		int y_max = y_min;

		for (int i = 1; i < points.size(); i++) {
			x_min = std::min(x_min, points[i].x);
			x_max = std::max(x_max, points[i].x);
			y_min = std::min(y_min, points[i].y);
			y_max = std::max(y_max, points[i].y);
		}

		cv::Rect rect;

		rect.x = x_min;
		rect.y = y_min;
		rect.width = x_max - x_min;
		rect.height = y_max - y_min;

		return rect;
	}

	/// <summary>
	/// Implements the Zhang-Suen line thinning algorithm.
	/// </summary>
	/// <param name="input"></param>
	/// <param name="output"></param>
	void thinningAlgorithm(cv::Mat& input, cv::Mat& output) {
		if (input.type() != CV_8UC1) {
			throw std::exception("This algorithm supports grayscale images only.");
		}

		output = input.clone();

		std::vector<bool> markers((uint64_t)input.rows * input.cols);

		bool repeat = true;
		int black_neighbours = 0;
		int white_black_transitions = 0;
		int a_count = 0;

		uint8_t region[9] = { 0 };

		for (int x = 0; x < output.cols; x++) {
			output.at<uint8_t>(0, x) = 255;
			output.at<uint8_t>(output.rows - 1, x) = 255;
		}

		for (int y = 0; y < output.rows; y++) {
			output.at<uint8_t>(y, 0) = 255;
			output.at<uint8_t>(y, output.cols - 1) = 255;
		}

		const int threshold = 128;

		for (int y = 1; y < output.rows - 1; y++) {
			for (int x = 1; x < output.cols - 1; x++) {
				uint8_t& value = output.at<uint8_t>(y, 0);

				value = value >= threshold ? 255 : 0;
			}
		}

		while (repeat) {
			repeat = false;
			white_black_transitions = 0;

			for (int i = 0; i < markers.size(); i++) {
				markers[i] = false;
			}

			for (int step = 0; step <= 1; step++) {
				// Two steps must be done.
				// The only differenece between them 

				for (int y = 1; y < output.rows - 1; y++) {
					for (int x = 1; x < output.cols - 1; x++) {
						region[0] = output.at<uint8_t>(y, x);

						if (region[0] == 255) {
							continue;
						}

						// 8 1 2
						// 7 0 3
						// 6 5 4
						// Maybe we can optimize this later?

						region[1] = output.at<uint8_t>(y - 1, x);
						region[2] = output.at<uint8_t>(y - 1, x + 1);
						region[3] = output.at<uint8_t>(y, x + 1);
						region[4] = output.at<uint8_t>(y + 1, x + 1);
						region[5] = output.at<uint8_t>(y + 1, x);
						region[6] = output.at<uint8_t>(y + 1, x - 1);
						region[7] = output.at<uint8_t>(y, x - 1);
						region[8] = output.at<uint8_t>(y - 1, x - 1);

						black_neighbours = 0;
						white_black_transitions = 0;

						for (int i = 1; i <= 8; i++) {
							if (region[i] == 0) {
								black_neighbours++;
							}

							int next = i == 8 ? 1 : i + 1;
							if (region[i] > region[next]) {
								white_black_transitions++;
							}
						}

						if (step == 0) {
							markers[x + (uint64_t) y * output.cols] =
								black_neighbours >= 2 && black_neighbours <= 6 &&
								white_black_transitions == 1 &&
								(region[1] || region[3] || region[5]) &&
								(region[3] || region[5] || region[7]);
						}
						else {
							markers[x + (uint64_t) y * output.cols] =
								black_neighbours >= 2 && black_neighbours <= 6 &&
								white_black_transitions == 1 &&
								(region[7] || region[1] || region[3]) &&
								(region[5] || region[7] || region[1]);
						}
					}
				}

				for (int y = 0; y < output.rows - 1; y++) {
					for (int x = 0; x < output.cols - 1; x++) {
						if (markers[x + (uint64_t)y * output.cols]) {
							output.at<uint8_t>(y, x) = 255;
							repeat = true;
						}
					}
				}
			}
		}
	}

	cv::Mat getRegionFeatures(cv::Mat& image, int dimension) {
		const uint8_t threshold = 127;

		cv::Mat regions = cv::Mat::zeros(dimension, dimension, CV_64F);
		cv::Mat region_totals = cv::Mat::zeros(dimension, dimension, CV_64F);

		int cell_width = (int) ceil((double)image.cols / dimension);
		int cell_height = (int) ceil((double)image.rows / dimension);

		for (int y = 0; y < image.rows; y++) {
			for (int x = 0; x < image.cols; x++) {
				auto value = image.at<uint8_t>(y, x);
				double& ref = regions.at<double>(y / cell_height, x / cell_width);

				if (value <= threshold) {
					ref += 1.0;
				}
			}
		}

		for (int y = 0; y < dimension; y++) {
			for (int x = 0; x < dimension; x++) {
				regions.at<double>(y, x) /= (double)cell_width * cell_height;
			}
		}

		return regions;
	}

	double getMappedDistance(const cv::Mat& ref, const cv::Mat& smpl)
	{
		if (ref.type() != smpl.type())
		{
			throw new std::exception("Matrices must have the same type.");
		}

		int type_used = ref.type();

		switch (type_used)
		{
			case CV_8UC1:
			case CV_64F:
				break;
			default:
				throw new std::exception("Matrices must either be CV_8UC1 or CV_64F");
		}

		double distance = 0.0;

		double ref_x = 0.0, ref_y = 0.0, smpl_x = 0.0, smpl_y = 0.0;

		double ref_advance_x, ref_advance_y, smpl_advance_x, smpl_advance_y;

		double ref_width = ref.size().width, ref_height = ref.size().height;
		double smpl_width = smpl.size().width, smpl_height = smpl.size().height;

		if (ref_width > smpl_width) {
			ref_advance_x = 1.0f;
			smpl_advance_x = smpl_width / ref_width;
		}
		else {
			ref_advance_x = ref_width / smpl_width;
			smpl_advance_x = 1.0f;
		}

		if (ref_height > smpl_height) {
			ref_advance_y = 1.0f;
			smpl_advance_y = smpl_height / ref_height;
		}
		else {
			ref_advance_y = ref_height / smpl_height;
			smpl_advance_y = 1.0f;
		}

		int count = 0;

		while (ref_y < ref_height && smpl_y < smpl_height) {
			double a, b;

			switch (type_used)
			{
				case CV_8UC1:
					a = ref.at<uint8_t>((int)ref_y, (int)ref_x);
					b = smpl.at<uint8_t>((int)smpl_y, (int)smpl_x);
					break;
				case CV_64F:
					a = ref.at<double>((int)ref_y, (int)ref_x);
					b = smpl.at<double>((int)smpl_y, (int)smpl_x);
					break;
			}

			double value = abs(a - b);
			distance += value * value;
			count++;

			ref_x += ref_advance_x;
			smpl_x += smpl_advance_x;

			if (ref_x >= ref_width || smpl_x >= smpl_width) {
				ref_x = 0.0;
				smpl_x = 0.0;

				ref_y += ref_advance_y;
				smpl_y += smpl_advance_y;
			}
		}

		distance /= count;

		return distance;
	}

	double getLetterDistance_Old(cv::Mat& ref, cv::Mat& smpl) {
		if (ref.type() != CV_64F || ref.type() != smpl.type() || ref.size() != smpl.size()) {
			throw new std::exception("Invalid input matrices!");
		}

		double distance = 0.0;

		uint64_t size = ref.size().width * ref.size().height;


		double* data_ref = ref.ptr<double>();
		double* data_smpl = smpl.ptr<double>();

		for (int i = 0; i < size; i++) {
			double value = abs(*data_ref - *data_smpl);

			distance += value * value;

			data_ref++;
			data_smpl++;
		}

		return distance;
	}
}