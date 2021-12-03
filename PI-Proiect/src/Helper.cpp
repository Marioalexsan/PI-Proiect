#include "Helper.hpp"

namespace pi {
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

		const double thresh = 0.2;

		bool bad_angles =
			abs(lineCos(a, b, c)) > thresh ||
			abs(lineCos(b, c, d)) > thresh ||
			abs(lineCos(c, d, a)) > thresh ||
			abs(lineCos(d, a, b)) > thresh;

		return !bad_angles;
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

	void pruneShort(std::vector<std::vector<cv::Point>>& target, double threshold) {
		for (int cindex = 0; cindex < target.size(); cindex++) {
			if (pi::contourPerimeter(target[cindex]) < threshold) {
				target.erase(target.begin() + cindex);
				cindex -= 1;
			}
		}
	}

	void compressContours(std::vector<std::vector<cv::Point>>& target) {
		int passes = 8;
		double start_threshold = 0.98;
		double relax_per_pass = 0.45 / passes;

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

	Rectangle getBoundingBox(std::vector<cv::Point>& points) {
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

		Rectangle rect;

		rect.x = x_min;
		rect.y = y_min;
		rect.width = x_max - x_min;
		rect.height = y_max - y_min;

		return rect;
	}
}