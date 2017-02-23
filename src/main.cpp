#include <iostream>
#include <chrono>
#include <opencv.hpp>

#include "ER.h"
#include "OCR.h"
#include "adaboost.h"
#include "utils.h"


using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	//get_canny_data();
	//train_classifier();
	//get_ocr_data(argc, argv, 0);
	//opencv_train();
	//train_cascade();
	//bootstrap();
	//rotate_image();
	//return 0;

	ERFilter* er_filter = new ERFilter(THRESHOLD_STEP, MIN_ER_AREA, MAX_ER_AREA, NMS_STABILITY_T, NMS_OVERLAP_COEF);
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

		ERs root;
		vector<ERs> pool;
		vector<ERs> strong;
		vector<ERs> weak;
		ERs tracked;
		vector<Text> text;

		er_filter->text_detect(frame, root, pool, strong, weak, tracked, text);

		end = chrono::high_resolution_clock::now();
		std::cout << "Running time: " << chrono::duration<double>(end - start).count() * 1000 << "ms\n";

		show_result(frame, pool, strong, weak, tracked, text);

		for (auto it : root)
			er_filter->er_delete(it);

		if (waitKey(1) >= 0) break;
	}

#else
	int img_count = 0;
	double time_sum = 0;
	vector<vector<Text>> det_text;
	for (int n = 1; n <= 233; n++)
	{
		Mat src;
		if (!load_test_file(src, n))	continue;
		else	img_count++;

		chrono::high_resolution_clock::time_point start, end;
		start = chrono::high_resolution_clock::now();

		ERs root;
		vector<ERs> pool;
		vector<ERs> strong;
		vector<ERs> weak;
		ERs tracked;
		vector<Text> text;

		er_filter->text_detect(src, root, pool, strong, weak, tracked, text);

		end = chrono::high_resolution_clock::now();
		time_sum += chrono::duration<double>(end - start).count();
		std::cout << "Running time: " << chrono::duration<double>(end - start).count() * 1000 << "ms\n";
		

		show_result(src, pool, strong, weak, tracked, text);
		det_text.push_back(text);
	}
	std::cout << "Average running time: " << 1000 * time_sum / img_count << "ms" << endl;

	save_deteval_xml(det_text);
#endif

	return 0;
}