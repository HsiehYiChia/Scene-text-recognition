#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>

#include "../inc/ER.h"
#include "../inc/OCR.h"
#include "../inc/adaboost.h"
#include "../inc/utils.h"


using namespace std;
using namespace cv;

int icdar_mode(ERFilter* er_filter);
int image_mode(ERFilter* er_filter, char filename[]);
int video_mode(ERFilter* er_filter, char filename[]);

int main(int argc, char* argv[])
{
	//get_lbp_data();
	//train_classifier();
	//train_ocr_model();
	//opencv_train();
	//train_cascade();
	//bootstrap();
	//rotate_ocr_samples();
	//draw_linear_time_MSER("res/ICDAR2015_test/img_7.jpg");
	//draw_multiple_channel("res/ICDAR2015_test/img_6.jpg");
	//test_MSER_time("res/ICDAR2015_test/img_7.jpg");
	//extract_ocr_sample();
	//calc_recall_rate();
	//test_best_detval();
	//make_video_ground_truth();
	//calc_video_result();
	//return 0;

	usage();

	ERFilter* er_filter = new ERFilter(THRESHOLD_STEP, MIN_ER_AREA, MAX_ER_AREA, NMS_STABILITY_T, NMS_OVERLAP_COEF, MIN_OCR_PROBABILITY);
	er_filter->stc = new CascadeBoost("er_classifier/strong.classifier");
	er_filter->wtc = new CascadeBoost("er_classifier/weak.classifier");
	er_filter->ocr = new OCR("ocr_classifier/OCR.model", OCR_IMG_L, OCR_FEATURE_L);
	er_filter->load_tp_table("dictionary/tp_table.txt");
	er_filter->corrector.load("dictionary/big.txt");

	char *filename = nullptr;
	if (strcmp(argv[1],"-icdar") == 0)
	{
		icdar_mode(er_filter);
	}
	else if(strcmp(argv[1], "-v") == 0)
	{
		if (argc == 3)
		{
			filename = argv[2];
		}
		video_mode(er_filter, filename);
	}
	else if (strcmp(argv[1], "-i") == 0)
	{
		if (argc == 3)
		{
			filename = argv[2];
		}
		image_mode(er_filter, filename);
	}
	else
	{
		cout << "Wrong input argument! " << endl;
		usage();
	}

	delete er_filter->wtc;
	delete er_filter->stc;
	delete er_filter->ocr;
	return 0;
}
