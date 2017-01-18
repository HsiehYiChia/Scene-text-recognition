#ifndef __MYLIB__
#define __MYLIB__

#include <iostream>
#include <chrono>
#include <opencv.hpp>

#include "ER.h"
#include "OCR.h"
#include "adaboost.h"


using namespace std;
using namespace cv;


bool load_test_file(Mat &src, int n);
void compute_channels(Mat &src, Mat &YCrcb, vector<Mat> &channels);


// for training
void get_canny_data();
void get_ocr_data(int argc, char **argv, int type);
void train_classifier();

#endif