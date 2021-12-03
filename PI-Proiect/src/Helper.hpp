#ifndef CONTOUR_PRUNE_HPP
#define CONTOUR_PRUNE_HPP

#include <vector>
#include <opencv2/core.hpp>

namespace pi {
	struct Rectangle {
		int x;
		int y;
		int width;
		int height;
	};

	double lineCos(cv::Point a, cv::Point b, cv::Point c);

	double contourPerimeter(const std::vector<cv::Point>& points);

	bool isLikeARectangle(const std::vector<cv::Point>& points);

	void pruneNonRectangles(std::vector<std::vector<cv::Point>>& target);

	void pruneShort(std::vector<std::vector<cv::Point>>& target, double threshold);

	void compressContours(std::vector<std::vector<cv::Point>>& target);

	void applyContrast(cv::Mat& img, float a, float b, float sa, float sb);

	Rectangle getBoundingBox(std::vector<cv::Point>& points);
}

#endif