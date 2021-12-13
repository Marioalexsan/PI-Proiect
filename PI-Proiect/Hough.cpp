#include"Hough.hpp"

hough_line::hough_line(cv::Mat& input, double theta, double r, int score) {
	int w = input.cols;
	int h = input.rows;

	//INFO: aici sa folosim un auxiliar?
	//cv::Mat auxiliary = input.clone();
	 
	double h_hough = (sqrt(2.0) * (double)(h > w ? h : w)) / 2.0;

	double center_x = w / 2.0;
	double center_y = h / 2.0;

	double h_sin = sin(theta);
	double h_cos = cos(theta);

	if (theta < PI*0.25 || theta > PI * 0.75) {
		double x1, y1, x2, y2;
		x1 = y1 = x2 = 0;
		y2 = h - 1;

		x1 = (
				((r - h_hough) - ((y1 - center_y) * h_sin)) / h_cos
			) + center_x;
		x2 = (
			((r - h_hough) - ((y2 - center_y) * h_sin)) / h_cos
			) + center_x;

		cv::Point p1(x1, y1), p2(x2, y2);
		cv::line(input, p1, p2, cv::Scalar(255, 0, 0));
	}
	else {
		double x1, y1, x2, y2;
		x1 = y1 = y2 = 0;
		x2 = w - 1;

		x1 = (
			((r - h_hough) - ((y1 - center_y) * h_cos)) / h_sin
			) + center_y;
		x2 = (
			((r - h_hough) - ((y2 - center_y) * h_cos)) / h_sin
			) + center_y;

		cv::Point p1(x1, y1), p2(x2, y2);
		cv::line(input, p1, p2, cv::Scalar(255, 0, 0));
	}
}

int hough_line::compare_score(hough_line h) {
	return (this->score - h->score);
}

void hough_transform(cv::Mat& input, cv::Mat& output);

void draw_lines_hough(cv::Mat& input, cv::Mat& output);
