#include "OCR.h"
#include "ER.h"

enum category
{
	big = 2,
	small = 1,
	change = 0
};
const char *table = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz&()";	// 10 num, 52 alphabet, 3 symbol and 1 '\0'
const int cat[65] = { big, big, big, big, big, big, big, big, big, big, 
big, big, change, big, big, big, big, big, big, change, big, big, big, big, change, change, big, big, change, big, change, change, change, change, big, change,
small, big, change, big, small, big, small, big, change, change, big, change, small, small, change, change, small, small, change, big, change, change, change, change, small, change,
big, big, big};




OCR::OCR(const char *svm_file_name)
{
	model = svm_load_model(svm_file_name);
}
OCR::~OCR()
{
	svm_free_model_content(model);
}



double OCR::lbp_run(vector<double> fv, const double angle)
{
	svm_node *node = new svm_node[fv.size() + 1];
	double *pv = new double[svm_get_nr_class(model)];

	int j = 0;
	for (int i = 0; i < fv.size(); i++)
	{
		if (fv[i] != 0)
		{
			node[j].index = i;
			node[j].value = fv[i]/129.0;
			j++;
		}
	}
	node[j].index = -1;
	

	const int label = svm_predict_probability(model, node, pv);
	const double prob = pv[label];
	cout << table[label] << " Probability = " << pv[label] << endl;
	
	delete[] node;
	delete[] pv;

	return table[label] + prob;
}


double OCR::chain_run(Mat &src, double angle)
{
	Mat ocr_img;

	//! pre process
	threshold(255 - src, ocr_img, 128, 255, CV_THRESH_OTSU);
	if (abs(angle) > 0.01)
		rotate_mat(ocr_img, ocr_img, angle, true);
	ARAN(ocr_img, ocr_img, 35);

	//! feature extract
	double *pv = new double[svm_get_nr_class(model)];
	svm_node *fv = new svm_node[201];
	extract_feature(ocr_img, fv);

	//! classify
	const int label = svm_predict_probability(model, fv, pv);
	const double prob = pv[label];

	//cout << table[label] << " Probability = " << pv[label] << endl;

	delete[] fv;
	delete[] pv;

	return table[label] + prob;
}



void OCR::extract_feature(Mat &src, svm_node *fv)
{
	Mat f_channel[8];
	for (int i = 0; i < 8; i++)
		f_channel[i] = Mat::zeros(35, 35, CV_8U);

	// get boundary direction
	vector<vector<Point>> contours;
	cv::findContours(src, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);


	// insert every boundary pixel into 8 bitmap
	for (int i = 0; i < contours.size(); i++)
	{
		contours[i].push_back(contours[i].front());
		for (int j = 0; j < contours[i].size() - 1; j++)
		{
			int dir = chain_code_direction(contours[i][j + 1], contours[i][j]);
			if (dir != -1)
				f_channel[dir].at<uchar>(contours[i][j]) = 255;
		}
	}


	// blur and normalize
	for (int i = 0; i < 8; i++)
	{
		GaussianBlur(f_channel[i], f_channel[i], Size(11, 11), 0);
		normalize(f_channel[i], f_channel[i], 0, 255, NORM_MINMAX, CV_8U);
		resize(f_channel[i], f_channel[i], Size(5, 5));

	}

	// make svm feature
	int j = 0;
	for (int i = 0; i < 8; i++)
	{
		uchar* ptr = f_channel[i].ptr<uchar>(0);
		for (int p = 0; p < 25; p++)
		{
			if (ptr[p] != 0)
			{
				fv[j].index = 25 * i + p;
				fv[j].value = ptr[p] / 255.0;
				j++;
			}
		}
	}
	fv[j].index = -1;

	/*for (int i = 0; i < 8; i++)
		resize(f_channel[i], f_channel[i], Size(100, 100));
	imshow("0", f_channel[0]);
	imshow("1", f_channel[1]);
	imshow("2", f_channel[2]);
	imshow("3", f_channel[3]);
	imshow("4", f_channel[4]);
	imshow("5", f_channel[5]);
	imshow("6", f_channel[6]);
	imshow("7", f_channel[7]);
	moveWindow("0", 50, 100);
	moveWindow("1", 200, 100);
	moveWindow("2", 350, 100);
	moveWindow("3", 500, 100);
	moveWindow("4", 650, 100);
	moveWindow("5", 800, 100);
	moveWindow("6", 950, 100);
	moveWindow("7", 1100, 100);
	imwrite("0.bmp", f_channel[0]);
	imwrite("1.bmp", f_channel[1]);
	imwrite("2.bmp", f_channel[2]);
	imwrite("3.bmp", f_channel[3]);
	imwrite("4.bmp", f_channel[4]);
	imwrite("5.bmp", f_channel[5]);
	imwrite("6.bmp", f_channel[6]);
	imwrite("7.bmp", f_channel[7]);
	waitKey(0);*/
}


// Don't use this function any more
void OCR::rotate_mat(Mat &src, Mat &dst, double angle, bool crop)
{
	double angle_rad = angle * CV_PI / 180;

	const int x0 = (src.cols - 1) / 2.0;
	const int y0 = (src.rows - 1) / 2.0;

	const int x1 = 0 - x0;
	const int y1 = 0 - y0;
	const int x2 = (src.cols - 1) - x0;
	const int y2 = 0 - y0;
	const int x3 = (src.cols - 1) - x0;
	const int y3 = (src.rows - 1) - y0;
	const int x4 = 0 - x0;
	const int y4 = (src.rows - 1) - y0;

	const int new_x1 = round(x1 * cos(angle_rad) - y1 * sin(angle_rad));
	const int new_y1 = round(x1 * sin(angle_rad) + y1 * cos(angle_rad));
	const int new_x2 = round(x2 * cos(angle_rad) - y2 * sin(angle_rad));
	const int new_y2 = round(x2 * sin(angle_rad) + y2 * cos(angle_rad));
	const int new_x3 = round(x3 * cos(angle_rad) - y3 * sin(angle_rad));
	const int new_y3 = round(x3 * sin(angle_rad) + y3 * cos(angle_rad));
	const int new_x4 = round(x4 * cos(angle_rad) - y4 * sin(angle_rad));
	const int new_y4 = round(x4 * sin(angle_rad) + y4 * cos(angle_rad));

	const int max_x = max(new_x1, max(new_x2, max(new_x3, new_x4)));
	const int max_y = max(new_y1, max(new_y2, max(new_y3, new_y4)));
	const int min_x = min(new_x1, min(new_x2, min(new_x3, new_x4)));
	const int min_y = min(new_y1, min(new_y2, min(new_y3, new_y4)));

	int crop_height = abs((max_x - min_x) * tan(angle_rad)) * 0.9;


	if (crop)
	{
		Mat tmp = Mat::zeros(max_y - min_y + 1 - 2 * crop_height, max_x - min_x + 1, CV_8U);
		for (int i = min_y; i < max_y; i++)
		{
			for (int j = min_x; j < max_x; j++)
			{
				double new_j = cos(angle_rad) *j - sin(angle_rad)*i + x0;
				double new_i = sin(angle_rad) *j + cos(angle_rad)*i + y0;


				if (new_i > 0 && new_j > 0 && new_i < src.rows - 1 && new_j < src.cols - 1 &&
					i >(min_y + crop_height) && i < (max_y - crop_height))
				{
					if (new_i == floor(new_i) && new_j == floor(new_j))
						tmp.at<uchar>(i - min_y, j - min_x) = src.at<uchar>(new_i, new_j);

					else
					{
						double alpha = new_i - floor(new_i);
						double beta = new_j - floor(new_j);
						uchar A = src.at<uchar>(new_i, new_j);
						uchar B = src.at<uchar>(new_i, new_j + 1);
						uchar C = src.at<uchar>(new_i + 1, new_j);
						uchar D = src.at<uchar>(new_i + 1, new_j + 1);
						uchar E = round((1 - alpha)*(1 - beta)*A + (1 - alpha)*beta*B + alpha*(1 - beta)*C + alpha*beta*D);

						tmp.at<uchar>(i - min_y - crop_height, j - min_x) = E;
					}
				}
			}
		}

		dst = tmp;
	}

	else
	{
		Mat tmp = Mat::zeros(max_y - min_y + 1, max_x - min_x + 1, CV_8U);
		for (int i = min_y; i < max_y; i++)
		{
			for (int j = min_x; j < max_x; j++)
			{
				double new_j = cos(angle_rad) *j - sin(angle_rad)*i + x0;
				double new_i = sin(angle_rad) *j + cos(angle_rad)*i + y0;


				if (new_i > 0 && new_j > 0 && new_i < src.rows - 1 && new_j < src.cols - 1)
				{
					if (new_i == floor(new_i) && new_j == floor(new_j))
						tmp.at<uchar>(i - min_y, j - min_x) = src.at<uchar>(new_i, new_j);

					else
					{
						double alpha = new_i - floor(new_i);
						double beta = new_j - floor(new_j);
						uchar A = src.at<uchar>(new_i, new_j);
						uchar B = src.at<uchar>(new_i, new_j + 1);
						uchar C = src.at<uchar>(new_i + 1, new_j);
						uchar D = src.at<uchar>(new_i + 1, new_j + 1);
						uchar E = round((1 - alpha)*(1 - beta)*A + (1 - alpha)*beta*B + alpha*(1 - beta)*C + alpha*beta*D);

						tmp.at<uchar>(i - min_y, j - min_x) = E;
					}
				}
			}
		}

		dst = tmp;
	}
	

}


void OCR::ARAN(Mat &src, Mat &dst, const int L, const double para)
{
	double R1 = (src.cols > src.rows) ? (double)src.rows / src.cols : (double)src.cols / src.rows;
	Size size_R2 = (src.cols > src.rows) ? Size(L, L * pow(R1, para)) : Size(L * pow(R1, para), L);

	Mat tmp;
	resize(src, tmp, size_R2, 0, 0, INTER_LINEAR);

	dst = Mat::zeros(L, L, CV_8U);
	if (tmp.cols > tmp.rows)
	{
		int offset = round((L - tmp.rows) / 2);
		for (int i = 0; i < tmp.rows; i++)
		{
			uchar* dptr = dst.ptr(i+offset);
			uchar* tptr = tmp.ptr(i);
			for (int j = 0; j < tmp.cols; j++)
			{
				dptr[j] = tptr[j];
			}
		}
	}

	else
	{
		int offset = round((L - tmp.cols) / 2);
		for (int i = 0; i < tmp.rows; i++)
		{
			uchar* dptr = dst.ptr(i, offset);
			uchar* tptr = tmp.ptr(i);
			for (int j = 0; j < tmp.cols; j++)
			{
				dptr[j] = tptr[j];
			}
		}
	}
}

int OCR::chain_code_direction(Point p1, Point p2)
{
	if (p1.x < p2.x && p1.y == p2.y)
		return 0;
	else if (p1.x < p2.x && p1.y < p2.y)
		return 1;
	else if (p1.x == p2.x && p1.y < p2.y)
		return 2;
	else if (p1.x > p2.x && p1.y < p2.y)
		return 3;
	else if (p1.x > p2.x && p1.y == p2.y)
		return 4;
	else if (p1.x > p2.x && p1.y > p2.y)
		return 5;
	else if (p1.x == p2.x && p1.y > p2.y)
		return 6;
	else if (p1.x < p2.x && p1.y > p2.y)
		return 7;
	return -1;
}

int OCR::index_mapping(char c)
{
	if (c >= '0' && c <= '9')
		return c - '0';
	else if (c >= 'A' && c <= 'Z')
		return c - 'A' + 10;
	else if (c >= 'a' && c <= 'z')
		return c - 'a' + 26 + 10;
	else if (c == '&')
		return 62;
	else if (c == '(')
		return 63;
	else if (c == ')')
		return 64;
	else
		return -1;
}