#ifndef HOUGH_HPP
#define HOUGH_HPP

/*
* ***********************************
*					Headers			*
*************************************
*/

#include <math.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


/*
* ***********************************
*					Defines			*
*************************************
*/

#define PI 3.141592653
#define MAXTHETA 180
#define STEP_MAXTHETA MAXTHETA/PI
#define NEIGHBOUR_SIZE 4

/*
* ***************************************
*					Class   			*
*****************************************
*/

class hough_line {
	double theta;
	double r;
	int score;

public:
	hough_line(cv::Mat& input, double theta, double r, int score);
	int compare_score(hough_line h);
};

/*
* ***************************************
*					Functions			*
*****************************************
*/
void hough_transform(cv::Mat& input, cv::Mat& output);

/*
* \brief
*	this function is drawing the lines that the Hough trnsform function created
*/
void draw_lines_hough(cv::Mat& input, cv::Mat& output);

#endif