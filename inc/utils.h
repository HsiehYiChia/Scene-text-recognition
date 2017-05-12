#ifndef __MYLIB__
#define __MYLIB__

#define VIDEO_MODE
//#define IMAGE_MODE

#define THRESHOLD_STEP 6
#define MIN_ER_AREA 150
#define MAX_ER_AREA 900000
#define NMS_STABILITY_T 2
#define NMS_OVERLAP_COEF 0.7
#define MIN_OCR_PROBABILITY 0.10
#define OCR_IMG_L 30
#define OCR_FEATURE_L 15
#define MAX_WIDTH 15000
#define MAX_HEIGHT 8000

#include <iostream>
#include <chrono>
#include <opencv.hpp>
#include <queue>
#include <fstream>
#include <string>
#include <sstream>

#include <time.h>
#include "ER.h"
#include "OCR.h"
#include "adaboost.h"


using namespace std;
using namespace std::chrono;
using namespace cv;


// Runtime Functions
bool load_challenge2_test_file(Mat &src, int n);
bool load_challenge2_training_file(Mat &src, int n);
void load_video_thread(VideoCapture &cap, Mat frame, Mat result, static vector<Text> *text, int *key);
void show_result(Mat& src, Mat& result_img, vector<Text> &text, vector<double> &times = vector<double>(), ERs &tracked = ERs(),
				vector<ERs> &strong = vector<ERs>(), vector<ERs> &weak = vector<ERs>(), vector<ERs> &all = vector<ERs>(), vector<ERs> &pool = vector<ERs>());
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
void save_deteval_xml(vector<vector<Text>> &text, string det_name = "det.xml");
void test_best_detval();
void make_video_ground_truth();
void calc_video_result();

// Training Functions
void get_lbp_data();
void bootstrap();
void rotate_ocr_samples();
void train_ocr_model();
void extract_ocr_sample();
void train_classifier();
void train_cascade();
void opencv_train();

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