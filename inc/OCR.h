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

#include <opencv.hpp>

#include "svm.h"

using namespace std;
using namespace cv;



class OCR
{
public:
	OCR() {};
	OCR(const char *svm_file_name);
	~OCR();
	double lbp_run(vector<double> fv, const double angle = 0);
	double chain_run(Mat &src, double angle);
	void rotate_mat(Mat &src, Mat &dst, double angle, bool crop = false);
	void ARAN(Mat &src, Mat &dst, const int L = 24, const double para = 0.5);
	void extract_feature(Mat &src, svm_node *fv);
	

private:
	svm_model *model;
	
	int index_mapping(char c);
	int chain_code_direction(Point p1, Point p2);
};

#endif