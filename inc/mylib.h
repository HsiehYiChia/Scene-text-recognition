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
void save_biggest_er(string inImg, string outfile);


// Training Function
void get_canny_data();
void get_ocr_data(int argc, char **argv, int type);
void train_classifier();
void train_cascade();
void opencv_train();
void save_pos_biggest_er();

#endif