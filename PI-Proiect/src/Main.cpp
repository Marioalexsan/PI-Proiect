#include <fstream>
#include <opencv2/highgui.hpp>

using namespace cv;

int main() {
	Mat img, imgGray;

	img = imread("texture.png");
	imshow("Inainte", img);

	imshow("Dupa", img);

	waitKey(0);

	return 0;
}