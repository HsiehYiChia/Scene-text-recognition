#ifndef __MYLIB__
#define __MYLIB__

//#define WEBCAM_MODE

#define THRESHOLD_STEP 2
#define MIN_ER_AREA 50
#define MAX_ER_AREA 90000
#define NMS_STABILITY_T 2
#define NMS_OVERLAP_COEF 0.7
#define MIN_OCR_PROBABILITY 0.002

#define MAX_WIDTH 1500
#define MAX_HEIGHT 800

#include <iostream>
#include <chrono>
#include <opencv.hpp>
#include <time.h>
#include "ER.h"
#include "OCR.h"
#include "adaboost.h"


using namespace std;
using namespace cv;


// Testing Functions
bool load_test_file(Mat &src, int n);
void show_result(Mat &src, vector<ERs> &all, vector<ERs> &pool, vector<ERs> &strong, vector<ERs> &weak, ERs &tracked, vector<Text> &text);
void draw_linear_time_MSER(string img_name);
void draw_multiple_channel(string img_name);
void test_MSER_time(string img_name);
void test_classifier_ROC(string classifier_name, string test_file);
vector<Vec4i> load_gt(int n);
Vec6d calc_detection_rate(int n, vector<Text> &text);
void save_deteval_xml(vector<vector<Text>> &text);


// Training Functions
void get_canny_data();
void bootstrap();
void rotate_ocr_samples();
void train_ocr_model();
void extract_ocr_sample();
void train_classifier();
void train_cascade();
void opencv_train();
#endif