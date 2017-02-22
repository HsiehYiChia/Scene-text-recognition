#ifndef __MYLIB__
#define __MYLIB__

#define WEBCAM_MODE

#define THRESHOLD_STEP 1
#define MIN_ER_AREA 30
#define MAX_ER_AREA 90000
#define NMS_STABILITY_T 2
#define NMS_OVERLAP_COEF 0.7

#define MAX_WIDTH 1500
#define MAX_HEIGHT 800

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
void show_result(Mat &src, vector<ERs> &pool, vector<ERs> &strong, vector<ERs> &weak, ERs &tracked, vector<Text> &text);
vector<Vec4i> load_gt(int n);
Vec6d calc_detection_rate(int n, vector<Text> &text);
void save_deteval_xml(vector<vector<Text>> &text);


// Training Functions
void get_canny_data();
void bootstrap();
void get_ocr_data(int argc, char **argv, int type);
void train_classifier();
void train_cascade();
void opencv_train();

#endif