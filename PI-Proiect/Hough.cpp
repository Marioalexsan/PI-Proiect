#include"Hough.hpp"

hough_line::hough_line(cv::Mat& input, double theta, double r = 0, int score = 0) {
	int w = input.cols;
	int h = input.rows;

	//INFO: aici sa folosim un auxiliar?
	//cv::Mat auxiliary = input.clone();
	 
	double h_hough = (sqrt(2.0) * (double)(h > w ? h : w)) / 2.0;

	double center_x = w / 2.0;
	double center_y = h / 2.0;

	double h_sin = sin(theta);
	double h_cos = cos(theta);


	this->r = r;
	this->score = score;
	this->theta = theta;


	if (theta < PI*0.25 || theta > PI * 0.75) {
		double x1, y1, x2, y2;
		x1 = y1 = x2 = 0;
		y2 = (double)h - 1;

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
		x2 = (double)w - 1;

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
	return (this->score - h.score);
}

std::vector<hough_line> hough_transform(cv::Mat& input, cv::Mat& output) {
	//width si height al imaginii
	int w = input.cols;
	int h = input.rows;

	//inaltime maxima pe care hough_arr trebuie sa o aiba
	const int h_hough = (sqrt(2.0) * (double)(h > w ? h : w)) / 2.0;

	//dublam inaltimea pentru procesarea r negativ
	const int double_hh = 2 * h_hough;

	//matricea hough
	int** hough_arr = new int* [h_hough];
	for (int i = 0; i < h_hough; i++) {
		hough_arr[i] = new int[double_hh];
		hough_arr[i] = 0;
	}

	//coordonatele centrului imaginii
	float center_X = w / 2;
	float center_Y = h / 2;

	//numarul de puncte care au fost adugate
	int added_points = 0;

	//valorile lui sin si cos pentru valorile theta
	double* sin_arr = new double[MAXTHETA];
	double* cos_arr = new double[MAXTHETA];

	for (int i = 0; i < MAXTHETA; ++i) {
		double aux = (double)i * STEP_MAXTHETA;
		sin_arr[i] = sin(aux);
		cos_arr[i] = cos(aux);
	}

	int r = 0;
	//adaug punctele
	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < h; ++j) {
			int newval0 = input.at<cv::Vec3b>(j, i)[0];
			if ((newval0 & 0x000000ff) != 0) {
				for (int k = 0; k < MAXTHETA; ++k) { //trec prin fiecare valoare a lui theta
					r = (int)(((double)i - center_X) * cos_arr[k]) + (((double)j - center_Y) * sin_arr[k]); //aflam valorile lui r
					r += h_hough; //pentru valorile negative
					if (r < 0) r = 0;
					if (r >= double_hh) r = double_hh;
					//INFO: aici nu pot sa scap de warning ;-;
					hough_arr[k][r]++;
				}
				added_points++;
			}
		}
	}

	//extragem liniile
	std::vector<hough_line> linii(20);

	if (added_points == 0) return linii;

	int threshold = 0;
	for (int i = 0; i < MAXTHETA; ++i) {
		int ok = 1;
		for (int r = NEIGHBOUR_SIZE; r < double_hh - NEIGHBOUR_SIZE; ++r) {
			if (hough_arr[i][r] > threshold) {
				int aux = hough_arr[i][r];

				//verific daca aux este maxim local
				for (int dx = -NEIGHBOUR_SIZE; dx <= NEIGHBOUR_SIZE; dx++) {
					for (int dy = -NEIGHBOUR_SIZE; dy <= NEIGHBOUR_SIZE; dy++) {
						int dt = i + dx;
						int dr = r + dy;
						if (dt < 0)
							dt = dt + MAXTHETA;
						else if (dt >= MAXTHETA)
							dt = dt - MAXTHETA;
						if (hough_arr[dt][dr] > aux) {
							// am gasit un alt maxim
							ok = 0;
						}
					}
				}

				if (ok == 0)
					continue;
				//calculam valoarea reala a lui theta
				double theta = i * STEP_MAXTHETA;

				//adaugam linia
				hough_line* linie = new hough_line(input, theta, r, hough_arr[i][r]);
				linii.push_back(linie);

			}
		}
	}
	return linii;
}

void draw_lines_hough(cv::Mat& input, cv::Mat& output);
