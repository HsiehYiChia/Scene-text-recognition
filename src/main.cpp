#include <iostream>
#include <chrono>
#include <opencv.hpp>

#include "ER.h"
#include "OCR.h"
#include "adaboost.h"
#include "mylib.h"


//#define DO_OCR

#define THRESHOLD_STEP 2
#define MIN_AREA 80
#define MAX_AREA 90000
#define STABILITY_T 1
#define OVERLAP_COEF 0.7
#define MIN_OCR_PROB 0.08


using namespace std;
using namespace cv;



int main(int argc, char** argv)
{
	//get_canny_data();
	//train_classifier();
	//get_ocr_data(argc, argv, 0);
	//opencv_train();
	train_cascade();
	return 0;

	ERFilter* er_filter = new ERFilter(THRESHOLD_STEP, MIN_AREA, MAX_AREA, STABILITY_T, OVERLAP_COEF);
	er_filter->adb1 = new CascadeBoost("er_classifier/cascade1.classifier");
	er_filter->adb2 = new CascadeBoost("er_classifier/cascade2.classifier");
	er_filter->adb3 = Algorithm::load<ml::Boost>("er_classifier/opencv_classifier.xml");
	er_filter->ocr = new OCR("ocr_classifier/OCR.model");
	er_filter->load_tp_table("transition_probability/tp.txt");

	double time_sum = 0;
	int k = 0;
	int l = 0;
	for (int n = 1; n <= 233; n++)
	{
		Mat src;
		if (!load_test_file(src, n))	continue;


		chrono::high_resolution_clock::time_point start, end;
		start = chrono::high_resolution_clock::now();


		Mat Ycrcb;
		vector<Mat> channel;
		compute_channels(src, Ycrcb, channel);

		ERs root(6);
		vector<ERs> pool(6);
		vector<ERs> strong(6);
		vector<ERs> weak(6);

#pragma omp parallel for
		for (int i = 0; i < channel.size(); i++)
		{
			root[i] = er_filter->er_tree_extract(channel[i]);
			er_filter->non_maximum_supression(root[i], pool[i], channel[i]);
			er_filter->classify(pool[i], strong[i], weak[i], channel[i]);
		}

		ERs all_er;
		er_filter->er_track(strong, weak, all_er, channel, Ycrcb);

#ifdef DO_OCR
		er_filter->er_grouping_ocr(all_er, channel, MIN_OCR_PROB, src.clone());
#else
		er_filter->er_grouping(all_er, src.clone());
#endif

		end = chrono::high_resolution_clock::now();
		std::cout << "Running time: " << chrono::duration<double>(end - start).count() * 1000 << "ms\n\n";
		time_sum += chrono::duration<double>(end - start).count();




		Mat strong_img = src.clone();
		Mat weak_img = src.clone();
		Mat all_img = src.clone();
		for (int i = 0; i < pool.size(); i++)
		{				
			for (auto it : weak[i])
				rectangle(weak_img, it->bound, Scalar(0, 0, 255));

			for (auto it : strong[i])
				rectangle(strong_img, it->bound, Scalar(0, 255, 0));

			for (auto it : pool[i])
				rectangle(all_img, it->bound, Scalar(255, 0, 0));
		}


		//imshow("src", src);
		imshow("weak", weak_img);
		imshow("strong", strong_img);
		//imshow("all", all_img);
		waitKey(0);
	}

	cout << "Average running time: " << 1000 * time_sum / 233.0 << "ms" << endl;
}