#ifndef CONTOUR_PRUNE_HPP
#define CONTOUR_PRUNE_HPP

/**************************************************************************************************/
/*                                           Headers                                              */
/**************************************************************************************************/
#include"Project_Headers.hpp"

#define BASE_VALUE 0


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

		void Run(const cv::Mat& input, cv::Mat& output);
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

	double lineCos(cv::Point a, cv::Point b, cv::Point c);

	double contourPerimeter(const std::vector<cv::Point>& points);

	/**
	 * \brief Function that verifies the conditions for a license plate
	 *
	 * \param[in] points - vector containing the four points of a rectangle
	 *
	 * \param[out] returns true if the contour fulfills the conditions
	 *
	 * \note
	 */
	bool isLikeALicensePlate(const std::vector<cv::Point>& points);

	double getColorMatch(cv::Mat& img, cv::Scalar color);

	void pruneNonRectangles(std::vector<std::vector<cv::Point>>& target);

	void pruneEmpty(std::vector<std::vector<cv::Point>>& target);

	void pruneShort(std::vector<std::vector<cv::Point>>& target, double threshold);

	/**
	 * \brief Function that simplifies the conturs provided by cv::findContours()
	 *
	 * \param[in] target - the image whose contours need to be simplified
	 *
	 * \note
	 */
	void simplifyContours(std::vector<std::vector<cv::Point>>& target, bool doLength = true);

	void applyContrast(cv::Mat& img, cv::Mat& output, float a, float b, float sa, float sb);

	cv::Rect getBoundingBox(std::vector<cv::Point>& points);

	void thinningAlgorithm(cv::Mat& input, cv::Mat& output);

	cv::Mat getRegionFeatures(cv::Mat& image, int dimension);

	/**
	 * \brief Function that calculates the distances between two images in the [0.0 ; 1.0] interval
	 *
	 * \param[in] ref - matrice of the first image
	 * 
	 * \param[in] smpl - matrice of the second image
	 *
	 * \param[out] distance - the distance between the two images
	 */
	double getImageDistance(const cv::Mat& ref, const cv::Mat& smpl);

	/**
	 * \brief Function that calculates the distances between two images in the [0.0 ; 1.0] interval
	 *
	 * \param[in] ref - matrice of the first image
	 *
	 * \param[in] smpl - matrice of the second image
	 * 
	 * \param[out] distance - the distance between the two images
	 *
	 * \note This is the simplified version of the pi::getImageDistance()
	 */
	double getLetterDistance_Old(cv::Mat& ref, cv::Mat& smpl);
}

#endif