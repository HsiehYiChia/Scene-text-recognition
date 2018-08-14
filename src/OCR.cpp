#include "../inc/OCR.h"
#include "../inc/ER.h"

enum category
{
	big = 2,
	small = 1,
	change = 0
};
const char *table = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz&()";	// 10 num, 52 alphabet, 3 symbol and 1 '\0'
const int cat[65] = { big, big, big, big, big, big, big, big, big, big, 
big, big, change, big, big, big, big, big, change, change, big, big, big, big, change, change, big, big, change, big, change, change, change, change, big, change,
small, big, change, big, small, big, small, big, change, change, big, change, small, small, change, change, small, small, change, big, change, change, change, change, small, change,
big, big, big};



OCR::OCR(const char *svm_file_name, int _img_L, int _feature_L) : img_L(_img_L), feature_L(_feature_L)
{
	model = svm_load_model(svm_file_name);
}


double OCR::lbp_run(Mat &src, int thresh, double slope)
{
	Mat ocr_img;

	threshold(255 - src, ocr_img, thresh, 255, cv::THRESH_OTSU);
	if (abs(slope) > 0.01)
	{
		double rad = atan2(slope, 1);
		//geometric_normalization(ocr_img, ocr_img, rad, false);
		rotate_mat(ocr_img, ocr_img, rad, true);
	}
	ARAN(ocr_img, ocr_img, img_L);

	ERFilter erfilter;
	Mat lbp = erfilter.calc_LBP(ocr_img);
	vector<double> fv = erfilter.make_LBP_hist(lbp, 2, img_L);

	svm_node *node = new svm_node[fv.size() + 1];

	int j = 0;
	for (int i = 0; i < fv.size(); i++)
	{
		if (fv[i] != 0)
		{
			node[j].index = i;
			node[j].value = fv[i] / 129.0;
			j++;
		}
	}
	node[j].index = -1;
	
	double *pv = new double[svm_get_nr_class(model)];
	const int label = svm_predict_probability(model, node, pv);
	const double prob = pv[label];
	cout << table[label] << " Probability = " << pv[label] << endl;
	
	delete[] node;
	delete[] pv;

	return table[label] + prob;
}


double OCR::chain_run(Mat src, int thresh, double slope)
{
	Mat ocr_img;

	//! pre process
	threshold(255-src, ocr_img, thresh, 255, cv::THRESH_OTSU);
	if (abs(slope) > 0.01)
	{
		double rad = atan2(slope, 1);
		//geometric_normalization(ocr_img, ocr_img, rad, false);
		rotate_mat(ocr_img, ocr_img, rad, true);
	}
	ARAN(ocr_img, ocr_img, img_L);

	/*imshow("input", src);
	imshow("rotated_ARAN", ocr_img);
	moveWindow("input", 200, 400);
	moveWindow("rotated_ARAN", 500, 400);*/

	//! feature extract
	svm_node *fv = new svm_node[8 * feature_L * feature_L + 1];
	extract_feature(ocr_img, fv);
	
	//! classify
	double *pv = new double[svm_get_nr_class(model)];
	const int label = svm_predict_probability(model, fv, pv);
	const double prob = pv[label];

	//cout << table[label] << " Probability = " << pv[label] << endl;

	/*Mat flann_feature = Mat::zeros(1, 8 * feature_L * feature_L, CV_32F);
	float *ptr = flann_feature.ptr<float>(0);
	int m = 0;
	while (fv[m].index != -1)
	{
		ptr[fv[m].index] = fv[m].value;
		m++;
	}
	vector<int> indice;
	vector<float> dists;
	index.knnSearch(flann_feature, indice, dists, 7, flann::SearchParams(64));
	vector<char> neighbors;

	for (int i = 0; i < indice.size(); i++)
	{
		neighbors.push_back(table[labels[indice[i]]]);
	}
	
	sort(neighbors.begin(), neighbors.end(), [](char a, char b) {return a < b; });
	int frequency = 0;
	int max_frequency = 0;
	char current_char = neighbors[0];
	char most_frequency = neighbors[0];
	for (int i = 0; i < neighbors.size(); i++)
	{
		if (neighbors[i] == current_char)
			frequency++;
		else if (frequency > max_frequency)
		{
			max_frequency = frequency;
			most_frequency = current_char;
		}
	}
	return most_frequency + max_frequency / 7.0;*/

	delete[] fv;
	delete[] pv;

	/*waitKey(0);
	destroyWindow("input");
	destroyWindow("rotated_ARAN");*/

	return table[label] + prob;
}



void OCR::extract_feature(Mat &src, svm_node *fv)
{
	/*imshow("src", src);
	moveWindow("src", 100, 500);*/

	Mat f_channel[8];
	for (int i = 0; i < 8; i++)
		f_channel[i] = Mat::zeros(img_L, img_L, CV_8U);

	// get boundary direction and insert every boundary pixel into 8 bitmap
	vector<vector<Point>> contours;
	cv::findContours(src, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() == 1)
			continue;
		else
			contours[i].push_back(contours[i].front());

		for (int j = 0; j < contours[i].size() - 1; j++)
		{
			int dir = chain_code_direction(contours[i][j + 1], contours[i][j]);
			f_channel[dir].at<uchar>(contours[i][j]) = 255;
		}
	}


	/*Mat all_chain_code = Mat::zeros(img_L, img_L, CV_8U);
	for (int i = 0; i < 8; i++)
	{
		all_chain_code += f_channel[i];
	}
	imshow("all", all_chain_code);
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
	moveWindow("7", 1100, 100);*/

	// blur and normalize
	for (int i = 0; i < 8; i++)
	{
		GaussianBlur(f_channel[i], f_channel[i], Size(7, 7), 0);
		normalize(f_channel[i], f_channel[i], 0, 255, NORM_MINMAX, CV_8U);
		resize(f_channel[i], f_channel[i], Size(feature_L, feature_L));
	}

	// make svm feature
	int j = 0;
	for (int i = 0; i < 8; i++)
	{
		uchar* ptr = f_channel[i].ptr<uchar>(0);
		for (int p = 0; p < feature_L*feature_L; p++)
		{
			if (ptr[p] != 0)
			{
				fv[j].index = feature_L * feature_L * i + p;
				fv[j].value = ptr[p] / 255.0;
				j++;
			}
		}
	}
	fv[j].index = -1;

	/*for (int i = 0; i < 8; i++)
		resize(f_channel[i], f_channel[i], Size(100, 100));
	imshow("0_gaussian", f_channel[0]);
	imshow("1_gaussian", f_channel[1]);
	imshow("2_gaussian", f_channel[2]);
	imshow("3_gaussian", f_channel[3]);
	imshow("4_gaussian", f_channel[4]);
	imshow("5_gaussian", f_channel[5]);
	imshow("6_gaussian", f_channel[6]);
	imshow("7_gaussian", f_channel[7]);
	moveWindow("0_gaussian", 50, 250);
	moveWindow("1_gaussian", 200, 250);
	moveWindow("2_gaussian", 350, 250);
	moveWindow("3_gaussian", 500, 250);
	moveWindow("4_gaussian", 650, 250);
	moveWindow("5_gaussian", 800, 250);
	moveWindow("6_gaussian", 950, 250);
	moveWindow("7_gaussian", 1100, 250);
	imwrite("0.bmp", f_channel[0]);
	imwrite("1.bmp", f_channel[1]);
	imwrite("2.bmp", f_channel[2]);
	imwrite("3.bmp", f_channel[3]);
	imwrite("4.bmp", f_channel[4]);
	imwrite("5.bmp", f_channel[5]);
	imwrite("6.bmp", f_channel[6]);
	imwrite("7.bmp", f_channel[7]);
	system("C:/Users/Larry/AppData/Local/Programs/Python/Python35/python.exe draw_chain_code.py");
	imshow("chain_code_direction", imread("chain_code_direction.png"));
	waitKey(0);
	destroyAllWindows();*/
}


// Don't use this function any more
void OCR::rotate_mat(Mat &src, Mat &dst, double rad, bool crop)
{
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

	const int new_x1 = round(x1 * cos(rad) - y1 * sin(rad));
	const int new_y1 = round(x1 * sin(rad) + y1 * cos(rad));
	const int new_x2 = round(x2 * cos(rad) - y2 * sin(rad));
	const int new_y2 = round(x2 * sin(rad) + y2 * cos(rad));
	const int new_x3 = round(x3 * cos(rad) - y3 * sin(rad));
	const int new_y3 = round(x3 * sin(rad) + y3 * cos(rad));
	const int new_x4 = round(x4 * cos(rad) - y4 * sin(rad));
	const int new_y4 = round(x4 * sin(rad) + y4 * cos(rad));

	const int max_x = max(new_x1, max(new_x2, max(new_x3, new_x4)));
	const int max_y = max(new_y1, max(new_y2, max(new_y3, new_y4)));
	const int min_x = min(new_x1, min(new_x2, min(new_x3, new_x4)));
	const int min_y = min(new_y1, min(new_y2, min(new_y3, new_y4)));


	if (crop)
	{
		int crop_height = (new_x2 - new_x1) * tan(rad) * 0.5;
		if (max_y - min_y + 1 - 2 * crop_height <= 0)
		{
			rotate_mat(src, dst, rad, false);
			return;
		}

		Mat tmp = Mat::zeros(max_y - min_y + 1 - 2 * crop_height, max_x - min_x + 1, CV_8U);
		for (int i = min_y+crop_height; i < max_y- crop_height; i++)
		{
			uchar *tptr = tmp.ptr(i - min_y- crop_height);
			for (int j = min_x; j < max_x; j++)
			{
				double new_j = cos(rad) *j - sin(rad)*(i-crop_height) + x0;
				double new_i = sin(rad) *j + cos(rad)*(i - crop_height) + y0;
				uchar *sptr = src.ptr(new_i, new_j);

				if (new_i > 0 && new_j > 0 && new_i < src.rows - 1 && new_j < src.cols - 1 &&
					i >(min_y + crop_height) && i < (max_y - crop_height))
				{
					if (new_i == floor(new_i) && new_j == floor(new_j))
						tptr[j - min_x] = sptr[0];

					else
					{
						double alpha = new_i - floor(new_i);
						double beta = new_j - floor(new_j);
						uchar A = sptr[0];
						uchar B = sptr[1];
						uchar C = sptr[src.cols];
						uchar D = sptr[src.cols + 1];
						tptr[j - min_x] = round((1 - alpha)*(1 - beta)*A + (1 - alpha)*beta*B + alpha*(1 - beta)*C + alpha*beta*D);
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
			uchar *tptr = tmp.ptr(i - min_y);
			for (int j = min_x; j < max_x; j++)
			{
				double new_j = cos(rad) *j - sin(rad)*i + x0;
				double new_i = sin(rad) *j + cos(rad)*i + y0;
				uchar *sptr = src.ptr(new_i, new_j);

				if (new_i > 0 && new_j > 0 && new_i < src.rows - 1 && new_j < src.cols - 1)
				{
					if (new_i == floor(new_i) && new_j == floor(new_j))
						tptr[j - min_x] = sptr[0];

					else
					{
						double alpha = new_i - floor(new_i);
						double beta = new_j - floor(new_j);
						uchar A = sptr[0];
						uchar B = sptr[1];
						uchar C = sptr[src.cols];
						uchar D = sptr[src.cols + 1];
						tptr[j - min_x] = round((1 - alpha)*(1 - beta)*A + (1 - alpha)*beta*B + alpha*(1 - beta)*C + alpha*beta*D);
					}
				}
			}
		}

		dst = tmp;
	}
	

}


// The code was based on the post:
// http://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
void OCR::geometric_normalization(Mat &src, Mat &dst, double rad, const bool crop)
{
	const double angle = rad * 180 / CV_PI;
	Point center(src.cols / 2, src.rows / 2);

	// scale the image if one of the side is bigger than 1000
	double scale = (src.cols > 1000 || src.rows > 1000) ? 500.0 / max(src.cols, src.rows) : 1.0;

	// get rotation matrix for rotating the image around its center
	Mat rot = getRotationMatrix2D(center, angle, scale);

	//determine the bounding box
	Rect bbox = RotatedRect(center, Size(src.cols*scale, src.rows*scale), angle).boundingRect();

	// adjust transformation matrix
	rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;

	if (crop)
	{
		const double crop_L = bbox.width * tan(rad) - 1;
		rot.at<double>(1, 2) -= crop_L;
		bbox.height -= 2 * crop_L;
	}

	warpAffine(src, dst, rot, bbox.size(), 1, 0, Scalar(255));
}


void OCR::ARAN(Mat &src, Mat &dst, const int L, const double para)
{
	double R1 = (src.cols > src.rows) ? (double)src.rows / src.cols : (double)src.cols / src.rows;
	Size size_R2 = (src.cols > src.rows) ? Size(L, L * pow(R1, para)) : Size(L * pow(R1, para), L);


	Mat tmp;
	resize(src, tmp, size_R2);
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



void OCR::feedback_verify(Text &text)
{
	// 1. See how many unchangeable upper-case letter and lower-case letter
	ERs big_letter;
	ERs small_letter;
	for (auto it : text.ers)
	{
		int idx = index_mapping(it->letter);
		if (cat[idx] == category::big)
			big_letter.push_back(it);
		else if (cat[idx] == category::small)
			small_letter.push_back(it);
	}

	// 2. Correct interchangeable letter
	for (auto it : text.ers)
	{
		const double T = 0.20;
		unsigned vote_big = 0;
		unsigned vote_little = 0;
		switch (it->letter)
		{
		case 'C': case 'J': case 'O': case 'P': case 'S': case 'U': case 'V': case 'W': case 'X': case 'Z':
			for (auto it2 : big_letter)
			{
				double offset = (it2->bound.x - it->bound.x) * text.slope;
				if (it->bound.y - it->bound.height*T > it2->bound.y - offset)
					vote_little++;
				else
					vote_big++;
			}
			for (auto it2 : small_letter)
			{
				double offset = (it2->bound.x - it->bound.x) * text.slope;
				if (it2->bound.y - it2->bound.height*T < it->bound.y + offset)
					vote_little++;
				else
					vote_big++;
			}
			it->letter = (vote_big > vote_little) ? it->letter : it->letter + 0x20;
			break;

		case 'c': case 'j': case 'o': case 'p': case 's': case 'u': case 'v': case 'w': case 'x': case 'z':
			if (!big_letter.empty())
			{
				for (auto it2 : big_letter)
				{
					double offset = (it2->bound.x - it->bound.x) * text.slope;
					if (it->bound.y - it->bound.height*T < it2->bound.y - offset)
						vote_big++;
					else
						vote_little++;
				}
			}
			else
			{
				for (auto it2 : small_letter)
				{
					double offset = (it2->bound.x - it->bound.x) * text.slope;
					if (it2->bound.y - it2->bound.height*T > it->bound.y + offset)
						vote_big++;
					else
						vote_little++;
				}
			}
			it->letter = (vote_big > vote_little) ? it->letter - 0x20 : it->letter;
			break;


		case '1': case 'i': case 'l':
			for (auto it2 : big_letter)
			{
				double offset = (it2->bound.x - it->bound.x) * text.slope;
				if (it->bound.y - it->bound.height*T > it2->bound.y - offset)
					vote_little++;
				else
					vote_big++;
			}
			for (auto it2 : small_letter)
			{
				double offset = (it2->bound.x - it->bound.x) * text.slope;
				if (it2->bound.y - it2->bound.height*T < it->bound.y + offset)
					vote_little++;
				else
					vote_big++;
			}
			if (vote_little > vote_big)
				it->letter = 'i';
			else
				it->letter = 'l';

			break;

		default:
			break;
		}
	}



	// 3. Count the amount of upper-case and amount of number character after first pass
	unsigned num_count = 0;
	unsigned big_count = 0;
	for (auto it : text.ers)
	{
		if (it->letter >= '0' && it->letter <= '9')
			num_count++;
		else if (it->letter >= 'A' && it->letter <= 'Z')
			big_count++;
	}


	// 4. Correct 'l' to 'I', 'O' to '0'
	for (auto it : text.ers)
	{
		if (big_count >= text.ers.size() / 2)
		{
			if (it->letter == 'l')
				it->letter = 'I';
		}
		if (num_count >= text.ers.size() / 2)
		{
			if (it->letter == 'O')
				it->letter = '0';
		}
	}



	text.word.clear();
	for (auto it : text.ers)
		text.word.append(string(1, it->letter));

	try_add_space(text);
}


void OCR::try_add_space(Text &text)
{
	vector<double> dist;
	for (int i = 0; i < text.ers.size() - 1; i++)
	{
		double d = text.ers[i + 1]->bound.x - text.ers[i]->bound.br().x;
		if (d >= 0)
		{
			dist.push_back(d);
		}
	}

	if (dist.empty())
		return;

	sort(dist.begin(), dist.end(), [](double a, double b) {return a < b; });
	double median_x_dist = dist[dist.size()/2];

	const double dist_thresh = 2.5;
	for (int i = text.ers.size() - 2; i >= 0; i--)
	{
		double d = text.ers[i + 1]->bound.x - text.ers[i]->bound.br().x;
		if (d > median_x_dist * dist_thresh && d > 0)
		{
			text.word.insert(i + 1, " ");
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