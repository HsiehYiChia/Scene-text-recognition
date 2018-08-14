#ifndef __MY_CV_LIB__
#define __MY_CV_LIB__

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <thread>
#include <chrono>
#include <numeric>
#include <omp.h>

#include <opencv2/opencv.hpp>

#include "svm.h"
#include "SpellingCorrector.h"

using namespace std;
using namespace cv;

struct Text;

class OCR
{
public:
	SpellingCorrector corrector;

	OCR() {};
	OCR(const char *svm_file_name, int _img_L, int _feature_L);
	~OCR() {};
	double lbp_run(Mat &src, int thresh, double slope = 0);		// use LBP spacial histogram as feature vector
	double chain_run(Mat src, int thresh, double slope = 0);	// use chain code as feature
	void feedback_verify(Text &text);
	void rotate_mat(Mat &src, Mat &dst, double rad, bool crop = false);
	void geometric_normalization(Mat &src, Mat &dst, double rad, const bool crop);
	void ARAN(Mat &src, Mat &dst, const int L = 24, const double para = 0.5);
	void extract_feature(Mat &src, svm_node *fv);
	int index_mapping(char c);
	

private:
	int img_L;
	int feature_L;
	svm_model *model;
	flann::Index index;
	Mat features;
	vector<int> labels;
	void try_add_space(Text &text);
	int chain_code_direction(Point p1, Point p2);
};
#endif