#include <iostream>
#include <chrono>
#include <opencv.hpp>

#include "ER.h"
#include "OCR.h"
#include "adaboost.h"
#include "mylib.h"


//#define DO_OCR

#define THRESHOLD_STEP 2
#define MIN_AREA 50
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
	//train_cascade();

	/*Mat haha = imread("res/pos2/2273.jpg");
	vector<Mat> Ycrcb;
	cvtColor(haha, haha, COLOR_BGR2YCrCb);
	split(haha, Ycrcb);
	ERFilter* f = new ERFilter(THRESHOLD_STEP, MIN_AREA, MAX_AREA, STABILITY_T, OVERLAP_COEF);

	Mat lbp = f->calc_LBP(Ycrcb[0], 24);
	Mat lbp_inv = f->calc_LBP(255- Ycrcb[0], 24);
	imshow("Y", lbp);
	imshow("Y inv", lbp_inv);

	lbp = f->calc_LBP(Ycrcb[1], 24);
	lbp_inv = f->calc_LBP(255 - Ycrcb[1], 24);
	imshow("Cr", lbp);
	imshow("Cr inv", lbp_inv);

	lbp = f->calc_LBP(Ycrcb[2], 24);
	lbp_inv = f->calc_LBP(255 - Ycrcb[2], 24);
	imshow("Cb", lbp);
	imshow("Cb inv", lbp_inv);

	moveWindow("Y", 100, 100);
	moveWindow("Y inv", 300, 100);
	moveWindow("Cr", 500, 100);
	moveWindow("Cr inv", 700, 100);
	moveWindow("Cb", 900, 100);
	moveWindow("Cb inv", 1100, 100);
	waitKey(0);*/
	//return 0;

	ERFilter* er_filter = new ERFilter(THRESHOLD_STEP, MIN_AREA, MAX_AREA, STABILITY_T, OVERLAP_COEF);
	er_filter->adb1 = new CascadeBoost("er_classifier/cascade1.classifier");
	er_filter->adb2 = new CascadeBoost("er_classifier/cascade2.classifier");
	er_filter->adb3 = Algorithm::load<ml::Boost>("er_classifier/opencv_classifier.xml");
	er_filter->ocr = new OCR("ocr_classifier/OCR.model");
	er_filter->load_tp_table("transition_probability/tp.txt");

	double time_sum = 0;
	int img_count = 0;
	for (int n = 1; n <= 400; n++)
	{
		Mat src;
		if (!load_test_file(src, n))
			continue;
		else
			img_count++;


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


		Mat group_result = src.clone();
		vector<Text> text;
#ifdef DO_OCR
		er_filter->er_grouping_ocr(all_er, channel, text, MIN_OCR_PROB, group_result);
#else
		er_filter->er_grouping(all_er, text);
#endif

		end = chrono::high_resolution_clock::now();
		std::cout << "Running time: " << chrono::duration<double>(end - start).count() * 1000 << "ms\n\n";

		time_sum += chrono::duration<double>(end - start).count();
		calc_detection_rate(n, text);

		Mat strong_img = src.clone();
		Mat weak_img = src.clone();
		Mat all_img = src.clone();
		Mat tracked = src.clone();
		for (int i = 0; i < pool.size(); i++)
		{				
			for (auto it : weak[i])
				rectangle(weak_img, it->bound, Scalar(0, 0, 255));

			for (auto it : strong[i])
				rectangle(strong_img, it->bound, Scalar(0, 255, 0));

			for (auto it : pool[i])
				rectangle(all_img, it->bound, Scalar(255, 0, 0));
		}

		for (auto it : all_er)
		{
			rectangle(tracked, it->bound, Scalar(255, 0, 255));
		}

		for (auto it : text)
		{
			rectangle(group_result, it.box, Scalar(0, 255, 255));
		}


		//imshow("src", src);
		//imshow("weak", weak_img);
		//imshow("strong", strong_img);
		imshow("all", all_img);
		imshow("tracked", tracked);
		imshow("group result", group_result);
		waitKey(0);
	}

	cout << "Average running time: " << 1000 * time_sum / img_count << "ms" << endl;
}