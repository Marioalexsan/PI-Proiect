#include"Hough.hpp"
hough_line::hough_line() {
	this->theta = 0;
	this->score = 0;
	this->r = 0;
	this->x1 = this->y1 = this->x2 = this->y2 = 0;
}
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
		//double x1, y1, x2, y2;
		this->x1 = this->y1 = this->x2 = 0;
		this->y2 = (double)h - 1;

		this->x1 = (
				((r - h_hough) - ((this->y1 - center_y) * h_sin)) / h_cos
			) + center_x;
		this->x2 = (
			((r - h_hough) - ((this->y2 - center_y) * h_sin)) / h_cos
			) + center_x;

		cv::Point p1(this->x1, this->y1), p2(this->x2, this->y2);
		cv::line(input, p1, p2, cv::Scalar(255, 0, 0));
	}
	else {
		//double x1, y1, x2, y2;
		this->x1 = this->y1 = this->y2 = 0;
		this->x2 = (double)w - 1;

		this->x1 = (
			((r - h_hough) - ((this->y1 - center_y) * h_cos)) / h_sin
			) + center_y;
		this->x2 = (
			((r - h_hough) - ((this->y2 - center_y) * h_cos)) / h_sin
			) + center_y;

		cv::Point p1(this->x1, this->y1), p2(this->x2, this->y2);
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
	const int h_hough = ((sqrt(2.0) * (double)(h > w ? h : w)) / 2.0);

	//dublam inaltimea pentru procesarea r negativ
	const int double_hh = 2 * h_hough;
	const int double_hw = 180;

	//matricea hough
	int* hough_arr = new int [h_hough];

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

	double r = 0;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			int newval = input.at<uchar>(y, x);
			if (newval >250) {
				for (int t = 0; t < MAXTHETA; ++t) {
					r = (((double)x - center_X) * cos_arr[t]) + (((double)y - center_Y) * sin_arr[t]);
					hough_arr[(int)((round(r + h_hough) * MAXTHETA)) + t]++;
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////
	/*
	std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines;  
4:    
5:            if(_accu == 0)  
6:                 return lines;  
7:    
8:            for(int r=0;r<_accu_h;r++)  
9:            {  
10:                 for(int t=0;t<_accu_w;t++)  
11:                 {  
12:                      if((int)_accu[(r*_accu_w) + t] >= threshold)  
13:                      {  
14:                           //Is this point a local maxima (9x9)  
15:                           int max = _accu[(r*_accu_w) + t];  
16:                           for(int ly=-4;ly<=4;ly++)  
17:                           {  
18:                                for(int lx=-4;lx<=4;lx++)  
19:                                {  
20:                                     if( (ly+r>=0 && ly+r<_accu_h) && (lx+t>=0 && lx+t<_accu_w) )  
21:                                     {  
22:                                          if( (int)_accu[( (r+ly)*_accu_w) + (t+lx)] > max )  
23:                                          {  
24:                                               max = _accu[( (r+ly)*_accu_w) + (t+lx)];  
25:                                               ly = lx = 5;  
26:                                          }  
27:                                     }  
28:                                }  
29:                           }  
30:                           if(max > (int)_accu[(r*_accu_w) + t])  
31:                                continue;  
32:    
33:    
34:                           int x1, y1, x2, y2;  
35:                           x1 = y1 = x2 = y2 = 0;  
36:    
37:                           if(t >= 45 && t <= 135)  
38:                           {  
39:                                //y = (r - x cos(t)) / sin(t)  
40:                                x1 = 0;  
41:                                y1 = ((double)(r-(_accu_h/2)) - ((x1 - (_img_w/2) ) * cos(t * DEG2RAD))) / sin(t * DEG2RAD) + (_img_h / 2);  
42:                                x2 = _img_w - 0;  
43:                                y2 = ((double)(r-(_accu_h/2)) - ((x2 - (_img_w/2) ) * cos(t * DEG2RAD))) / sin(t * DEG2RAD) + (_img_h / 2);  
44:                           }  
45:                           else  
46:                           {  
47:                                //x = (r - y sin(t)) / cos(t);  
48:                                y1 = 0;  
49:                                x1 = ((double)(r-(_accu_h/2)) - ((y1 - (_img_h/2) ) * sin(t * DEG2RAD))) / cos(t * DEG2RAD) + (_img_w / 2);  
50:                                y2 = _img_h - 0;  
51:                                x2 = ((double)(r-(_accu_h/2)) - ((y2 - (_img_h/2) ) * sin(t * DEG2RAD))) / cos(t * DEG2RAD) + (_img_w / 2);  
52:                           }  
53:    
54:                           lines.push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));  
55:    
56:                      }  
57:                 }  
58:            }  
59:    
60:            std::cout << "lines: " << lines.size() << " " << threshold << std::endl;  
61:            return lines;  
	*/
	///////////////////////////////////////////////////////////////////////////////
	
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
						if (hough_arr[dt * w + dr] > aux) {
							// am gasit un alt maxim
							ok = 0;
						}
					}
				}

				if (ok == 0)
					continue;
				//calculam valoarea reala a lui theta
				double theta = (double)i * STEP_MAXTHETA;

				//adaugam linia
				hough_line linie = hough_line(input, theta, r, hough_arr[i][r]);
				linii.push_back(linie);

			}
		}
	}
	return linii;
}

void draw_lines_hough(cv::Mat& input, cv::Mat& output, std::vector<hough_line> linii) {
	if (linii.size() != 0) {
		for (std::vector<hough_line>::iterator it = linii.begin(); it != linii.end(); ++it) {
			hough_line h = *it;
			cv::Point p1(h.x1, h.y1), p2(h.x2, h.y2);
			cv::line(input, p1, p2, cv::Scalar(255, 0, 0));
		}
	}
}
