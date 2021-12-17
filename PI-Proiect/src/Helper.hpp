#ifndef CONTOUR_PRUNE_HPP
#define CONTOUR_PRUNE_HPP

#include <vector>
#include <opencv2/core.hpp>

namespace pi {
	class OperationList {
	private:

		std::vector<std::function<void(cv::Mat&, cv::Mat&)>> steps;

	public:

		OperationList();
		OperationList(OperationList&) = delete;
		OperationList(OperationList&&) = delete;

		~OperationList();

		void AddStep(std::function<void(cv::Mat&, cv::Mat&)> step);

		void Clear();

		void Run(cv::Mat& input, cv::Mat& output);
	};

	struct Rectangle {
		int x;
		int y;
		int width;
		int height;
	};

	double lineCos(cv::Point a, cv::Point b, cv::Point c);

	double contourPerimeter(const std::vector<cv::Point>& points);

	bool isLikeARectangle(const std::vector<cv::Point>& points);

	double getColorMatch(cv::Mat& img, cv::Scalar color);

	void pruneNonRectangles(std::vector<std::vector<cv::Point>>& target);

	void pruneEmpty(std::vector<std::vector<cv::Point>>& target);

	void pruneShort(std::vector<std::vector<cv::Point>>& target, double threshold);

	void simplifyContours(std::vector<std::vector<cv::Point>>& target);

	void applyContrast(cv::Mat& img, float a, float b, float sa, float sb);

	Rectangle getBoundingBox(std::vector<cv::Point>& points);

	void thinningAlgorithm(cv::Mat& input, cv::Mat& output);

	cv::Mat getRegionFeatures(cv::Mat& image, int dimension);

	double getLetterDistance(cv::Mat& ref, cv::Mat& smpl);
}

#endif