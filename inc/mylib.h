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


// Testing Functions
bool load_test_file(Mat &src, int n);
void compute_channels(Mat &src, Mat &YCrcb, vector<Mat> &channels);
vector<Vec4i> load_gt(int n);
Vec6d calc_detection_rate(int n, vector<Text> &text);
void save_deteval_xml(vector<vector<Text>> &text);


// Training Functions
void get_canny_data();
void get_ocr_data(int argc, char **argv, int type);
void train_classifier();
void train_cascade();
void opencv_train();




#endif