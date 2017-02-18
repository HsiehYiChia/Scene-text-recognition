#include <iostream>
#include <chrono>
#include <opencv.hpp>

#include "ER.h"
#include "OCR.h"
#include "adaboost.h"
#include "mylib.h"

//#define WEBCAM_MODE
//#define DO_OCR

#define THRESHOLD_STEP 5
#define MIN_AREA 10
#define MAX_AREA 90000
#define STABILITY_T 1
#define OVERLAP_COEF 0.7
#define MIN_OCR_PROB 0.08


using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	get_canny_data();
	//train_classifier();
	//get_ocr_data(argc, argv, 0);
	//opencv_train();
	//train_cascade();
	//bootstrap();
	return 0;

	ERFilter* er_filter = new ERFilter(THRESHOLD_STEP, MIN_AREA, MAX_AREA, STABILITY_T, OVERLAP_COEF);
	er_filter->adb1 = new CascadeBoost("er_classifier/cascade1.classifier");
	er_filter->adb2 = new CascadeBoost("er_classifier/cascade2.classifier");
	er_filter->ocr = new OCR("ocr_classifier/OCR.model");
	er_filter->load_tp_table("transition_probability/tp.txt");

#ifdef WEBCAM_MODE
	VideoCapture cap(0);
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	for (;;)
	{
		Mat frame;
		cap >> frame;

		chrono::high_resolution_clock::time_point start, end;
		start = chrono::high_resolution_clock::now();

		Mat Ycrcb;
		vector<Mat> channel;
		compute_channels(frame, Ycrcb, channel);

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

		vector<Text> text;
	#ifdef DO_OCR
		er_filter->er_grouping_ocr(all_er, channel, text, MIN_OCR_PROB);
	#else
		er_filter->er_grouping(all_er, text);
	#endif

		
		
		end = chrono::high_resolution_clock::now();
		std::cout << "Running time: " << chrono::duration<double>(end - start).count() * 1000 << "ms\n";

		
		Mat strong_img = frame.clone();
		Mat weak_img = frame.clone();
		Mat all_img = frame.clone();
		Mat tracked = frame.clone();
		Mat group_result = frame.clone();
		for (int i = 0; i < pool.size(); i++)
		{
			for (auto it : weak[i])
				rectangle(weak_img, it->bound, Scalar(0, 0, 255));

			for (auto it : strong[i])
				rectangle(strong_img, it->bound, Scalar(0, 255, 0));
		}

		for (auto it : all_er)
		{
			rectangle(tracked, it->bound, Scalar(255, 0, 255));
		}

		for (auto it : text)
		{
			rectangle(group_result, it.box, Scalar(0, 255, 255));
		}


		for (int i = 0; i < root.size(); i++)
		{
			er_filter->er_delete(root[i]);
		}
		
		imshow("weak", weak_img);
		imshow("strong", strong_img);
		imshow("tracked", tracked);
		imshow("group result", group_result);
		if (waitKey(1) >= 0) break;
	}
	return 0;

#else
	int img_count = 0;
	double time_sum = 0;
	double tp = 0;
	double GT = 0;
	double detect = 0;
	double recall = 0;
	double precision = 0;
	vector<vector<Text>> det_text;
	for (int n = 1; n <= 233; n++)
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


		
		vector<Text> text;
	#ifdef DO_OCR
		er_filter->er_grouping_ocr(all_er, channel, text, MIN_OCR_PROB);
	#else
		er_filter->er_grouping(all_er, text);
	#endif

		end = chrono::high_resolution_clock::now();
		Vec6d rate = calc_detection_rate(n, text);
		std::cout << "Running time: " << chrono::duration<double>(end - start).count() * 1000 << "ms\n";
		std::cout << "Recall: " << rate[3] << "    Precision: " << rate[4] << "    Hmean: " << rate[5] << endl << endl;
		
		
		
		// calculate average time, recall, precision
		time_sum += chrono::duration<double>(end - start).count();
		tp += rate[0];
		GT += rate[1];
		detect += rate[2];
		recall += rate[3];
		precision += rate[4];
		
		
		
		Mat strong_img = src.clone();
		Mat weak_img = src.clone();
		Mat all_img = src.clone();
		Mat tracked = src.clone();
		Mat group_result = src.clone();
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

		det_text.push_back(text);

		//imshow("src", src);
		imshow("weak", weak_img);
		imshow("strong", strong_img);
		imshow("all", all_img);
		imshow("tracked", tracked);
		imshow("group result", group_result);
		waitKey(0);
	}

	recall = tp / GT;
	precision = tp / detect;
	//recall /= img_count;
	//precision /= img_count;
	
	std::cout << "Average running time: " << 1000 * time_sum / img_count << "ms" << endl;
	std::cout << "Recall: " << recall << endl;
	std::cout << "Precision: " << precision << endl;
	std::cout << "Harmonic mean: " << (recall * precision * 2) / (recall + precision) << endl;

	save_deteval_xml(det_text);
#endif
	return 0;
}