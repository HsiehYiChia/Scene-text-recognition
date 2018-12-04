#ifndef __MYLIB__
#define __MYLIB__

#define POS 1
#define NEG -1
#define THRESHOLD_STEP 8
#define MIN_ER_AREA 120
#define MAX_ER_AREA 900000
#define NMS_STABILITY_T 2
#define NMS_OVERLAP_COEF 0.7
#define MIN_OCR_PROBABILITY 0.15
#define OCR_IMG_L 30
#define OCR_FEATURE_L 15
#define MAX_WIDTH 15000
#define MAX_HEIGHT 8000
#define MAX_FILE_PATH 100
#define MAX_FILE_NUMBER 50000

/* for libsvm */
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <queue>
#include <fstream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
//#include <filesystem>


#include <time.h>
#include "ER.h"
#include "OCR.h"
#include "adaboost.h"


using namespace std;
using namespace std::chrono;
using namespace cv;

// info function
void usage();
void print_result(int img_count, vector<double> avg_time);

// getopt function
int image_mode(ERFilter* er_filter, char filename[]);
int video_mode(ERFilter* er_filter, char filename[]);

// Runtime Functions
bool load_challenge2_test_file(Mat &src, int n);
bool load_challenge2_training_file(Mat &src, int n);
void load_video_thread(VideoCapture &cap, Mat frame, Mat result, vector<Text> *text, int *key);
void show_result(Mat& src, Mat& result_img, vector<Text> &text, vector<double> times = vector<double>(), ERs tracked = ERs(),
				vector<ERs> strong = vector<ERs>(), vector<ERs> weak = vector<ERs>(), vector<ERs> all = vector<ERs>(), vector<ERs> pool = vector<ERs>());
void draw_FPS(Mat& src, double time);


// Testing Functions
void draw_linear_time_MSER(string img_name);
void draw_multiple_channel(string img_name);
void output_MSER_time(string img_name);
void output_classifier_ROC(string classifier_name, string test_file);
void output_optimal_path(string img_name);
vector<Vec4i> load_gt(int n);
Vec6d calc_detection_rate(int n, vector<Text> &text);	// Deprecated
void calc_recall_rate();
void save_deteval_xml(vector<vector<Text>> &text, string det_name);
void test_best_detval();
void make_video_ground_truth();
void calc_video_result();

// Training Functions
void get_lbp_data();
void get_ocr_data();
void train_ocr_model();
void train_detection_classifier();
void bootstrap();
void rotate_ocr_samples();

void extract_ocr_sample();

// solve levenshtein distance(edit distance) by dynamic programming, 
// check https://vinayakgarg.wordpress.com/2012/12/10/edit-distance-using-dynamic-programming/ for more info
int levenshtein_distance(string str1, string str2);


class Profiler
{
public:
	Profiler();
	void Start();
	int Count();
	double Stop();
	void Log(std::string name);
	void Message(std::string msg, float value);
	void Report();

protected:
	int count;
	std::chrono::time_point<std::chrono::steady_clock> time;
	struct record
	{
		record(std::string _name, long long _duration);
		record(std::string _name, float _value, bool _is_msg);
		std::string name;
		long long duration;
		float value;
		bool is_msg;
	};
	std::queue<record> logs;
};

#endif
