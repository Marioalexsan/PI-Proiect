#ifndef CONTOUR_PRUNE_HPP
#define CONTOUR_PRUNE_HPP


#include"Project_Headers.hpp"

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

	/**************************************************************************************************/
	/*                                     Public Functions                                          */
	/**************************************************************************************************/

	/**
	 * \brief Function that extracts the regions from Mittelschrift_regions.txt
	 *
	 * \param[in] path - the path for the file to be segmented
	 * 
	 * \param[out] result - an unordered map in which char is the letter from the image and Rect is the rectangular region bountig the char
	 *
	 * \note The letter's regions are used for the letters segmentation
	 */
	std::unordered_map<char, cv::Rect> loadLetterRectangles(std::string path);

	/**
	 * \brief 
	 *
	 * \param[in] 
	 *
	 * \param[out] 
	 *
	 * \note 
	 */
	double lineCos(cv::Point a, cv::Point b, cv::Point c);

	/**
	 * \brief
	 *
	 * \param[in]
	 *
	 * \param[out]
	 *
	 * \note
	 */
	double contourPerimeter(const std::vector<cv::Point>& points);

	/**
	 * \brief
	 *
	 * \param[in]
	 *
	 * \param[out]
	 *
	 * \note
	 */
	bool isLikeARectangle(const std::vector<cv::Point>& points);

	/**
	 * \brief
	 *
	 * \param[in]
	 *
	 * \param[out]
	 *
	 * \note
	 */
	double getColorMatch(cv::Mat& img, cv::Scalar color);

	/**
	 * \brief 
	 *
	 * \param[in] 
	 *
	 * \param[out] 
	 *
	 * \note 
	 */
	void pruneNonRectangles(std::vector<std::vector<cv::Point>>& target);

	/**
	 * \brief
	 *
	 * \param[in]
	 *
	 * \param[out]
	 *
	 * \note
	 */
	void pruneEmpty(std::vector<std::vector<cv::Point>>& target);

	/**
	 * \brief
	 *
	 * \param[in]
	 *
	 * \param[out]
	 *
	 * \note
	 */
	void pruneShort(std::vector<std::vector<cv::Point>>& target, double threshold);

	/**
	 * \brief
	 *
	 * \param[in]
	 *
	 * \param[out]
	 *
	 * \note
	 */
	void simplifyContours(std::vector<std::vector<cv::Point>>& target);

	/**
	 * \brief
	 *
	 * \param[in]
	 *
	 * \param[out]
	 *
	 * \note
	 */
	void applyContrast(cv::Mat& img, float a, float b, float sa, float sb);

	/**
	 * \brief
	 *
	 * \param[in]
	 *
	 * \param[out]
	 *
	 * \note
	 */
	Rectangle getBoundingBox(std::vector<cv::Point>& points);

	/**
	 * \brief
	 *
	 * \param[in]
	 *
	 * \param[out]
	 *
	 * \note
	 */
	void thinningAlgorithm(cv::Mat& input, cv::Mat& output);

	/**
	 * \brief
	 *
	 * \param[in]
	 *
	 * \param[out]
	 *
	 * \note
	 */
	cv::Mat getRegionFeatures(cv::Mat& image, int dimension);

	/**
	 * \brief
	 *
	 * \param[in]
	 *
	 * \param[out]
	 *
	 * \note
	 */
	double getMappedDistance(cv::Mat& ref, cv::Mat& smpl);

	/**
	 * \brief
	 *
	 * \param[in]
	 *
	 * \param[out]
	 *
	 * \note
	 */
	double getLetterDistance_Old(cv::Mat& ref, cv::Mat& smpl);
}

#endif