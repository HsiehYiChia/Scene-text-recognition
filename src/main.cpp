#include <iostream>
#include <chrono>
#include <thread>
#include <opencv.hpp>

#include "ER.h"
#include "OCR.h"
#include "adaboost.h"
#include "utils.h"


using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	//get_lbp_data();
	//train_classifier();
	//train_ocr_model();
	//opencv_train();
	//train_cascade();
	//bootstrap();
	//rotate_ocr_samples();
	//draw_linear_time_MSER("img_7.jpg");
	//draw_multiple_channel("img_6.jpg");
	//test_MSER_time("img_7.jpg");
	//extract_ocr_sample();
	//calc_recall_rate();
	//test_best_detval();
	//make_video_ground_truth();
	//calc_video_result();
	//return 0;

	ERFilter* er_filter = new ERFilter(THRESHOLD_STEP, MIN_ER_AREA, MAX_ER_AREA, NMS_STABILITY_T, NMS_OVERLAP_COEF, MIN_OCR_PROBABILITY);
	er_filter->stc = new CascadeBoost("er_classifier/cascade1.classifier");
	er_filter->wtc = new CascadeBoost("er_classifier/weak.classifier");
	er_filter->ocr = new OCR("ocr_classifier/OCR.model", OCR_IMG_L, OCR_FEATURE_L);
	er_filter->load_tp_table("tp_table.txt");
	er_filter->corrector.load("dictionary/modified_big.txt");
	er_filter->corrector.load("dictionary/self_define_word.txt");

#if defined(VIDEO_MODE)
	//VideoCapture cap(0);
	VideoCapture cap("video_result/result4/test2.mpg");
	if (!cap.isOpened())
	{
		cerr << "ERROR! Unable to open camera or video file\n";
		return -1;
	}
		
	Mat frame;
	VideoWriter writer;
	cap >> frame;	// get 1 frame to know the frame size
	writer.open("video_result/result/result.wmv", CV_FOURCC('W', 'M', 'V', '2'), 20.0, frame.size(), true);
	if (!writer.isOpened()) {
		cerr << "Could not open the output video file for write\n";
		return -1;
	}
	
	Mat result;
	static vector<Text> result_text;
	int key = -1;
	//thread capture_thread(load_video_thread, cap, frame, result, &result_text, &key);

	chrono::high_resolution_clock::time_point start, end;
	start = chrono::high_resolution_clock::now();
	const int frame_count = 2;
	int img_count = 0;
	vector<double> avg_time(7, 0);
	fstream f_result_text("video_result/result/det.txt", fstream::out);

	for (;;)
	{
		ERs tracked_vec;
		ERs root_vec;
		vector<Mat> channel_vec;

		for (int n = 0; n < frame_count; n++)
		{
			cap >> frame;
			if (frame.empty())	break;

			Mat Ycrcb;
			vector<Mat> channel;
			er_filter->compute_channels(frame, Ycrcb, channel);

			if (n == frame_count / 2)
				channel_vec = channel;

			ERs root(channel.size());
			vector<ERs> all(channel.size());
			vector<ERs> pool(channel.size());
			vector<ERs> strong(channel.size());
			vector<ERs> weak(channel.size());
			ERs tracked;
			vector<chrono::high_resolution_clock::time_point> time_vec(channel.size() * 4 + 2);

		#pragma omp parallel for
			for (int i = 0; i < channel.size(); i++)
			{
				time_vec[i * 4] = chrono::high_resolution_clock::now();
				root[i] = er_filter->er_tree_extract(channel[i]);
				time_vec[i * 4 + 1] = chrono::high_resolution_clock::now();
				er_filter->non_maximum_supression(root[i], all[i], pool[i], channel[i]);
				time_vec[i * 4 + 2] = chrono::high_resolution_clock::now();
				er_filter->classify(pool[i], strong[i], weak[i], channel[i]);
				time_vec[i * 4 + 3] = chrono::high_resolution_clock::now();
			}
			time_vec.rbegin()[1] = chrono::high_resolution_clock::now();
			er_filter->er_track(strong, weak, tracked, channel, Ycrcb);
			time_vec.rbegin()[0] = chrono::high_resolution_clock::now();


			// push this frame's root and tracked into accumulate result
			root_vec.insert(root_vec.end(), root.begin(), root.end());
			tracked_vec.insert(tracked_vec.end(), tracked.begin(), tracked.end());

			// calculate time of each module, maximum value will be selected in parallel section 
			chrono::duration<double> extract_time;
			chrono::duration<double> nms_time;
			chrono::duration<double> classify_time;
			for (int i = 0; i < channel.size(); i++)
			{
				if (extract_time < time_vec[i * 4 + 1] - time_vec[i * 4])
					extract_time = time_vec[i * 4 + 1] - time_vec[i * 4];

				if (nms_time < time_vec[i * 4 + 2] - time_vec[i * 4 + 1])
					nms_time = time_vec[i * 4 + 2] - time_vec[i * 4 + 1];

				if (classify_time < time_vec[i * 4 + 3] - time_vec[i * 4 + 2])
					classify_time = time_vec[i * 4 + 3] - time_vec[i * 4 + 2];
			}
			chrono::duration<double> track_time = (time_vec.rbegin()[2] - time_vec.rbegin()[3]);
			avg_time[0] += extract_time.count();
			avg_time[1] += nms_time.count();
			avg_time[2] += classify_time.count();
			avg_time[3] += track_time.count();

			show_result(frame, result, result_text);

			// write file
			char buf[60];
			sprintf(buf, "video_result/result/%d.jpg", img_count);
			imwrite(buf, result);
			writer << result;
			sort(result_text.begin(), result_text.end(), [](Text a, Text b) {return a.box.y < b.box.y; });
			f_result_text << img_count;
			for (auto it : result_text)
			{
				f_result_text << "," << it.word;
			}
			f_result_text << endl;
			++img_count;

			// check key press
			key = waitKey(1);
			if (key >= 0) break;
		}
		if (key >= 0 || frame.empty()) break;

		vector<chrono::high_resolution_clock::time_point> time(3);

		vector<Text> tmp_text;
	#ifndef DO_OCR
		time[0] = chrono::high_resolution_clock::now();
		er_filter->er_grouping(tracked_vec, text, true, true);
		time[1] = chrono::high_resolution_clock::now();
	#else
		time[0] = chrono::high_resolution_clock::now();
		er_filter->er_grouping(tracked_vec, tmp_text, false, true);
		time[1] = chrono::high_resolution_clock::now();
		er_filter->er_ocr(tracked_vec, channel_vec, tmp_text);
		time[2] = chrono::high_resolution_clock::now();
	#endif
		result_text = tmp_text;

		chrono::duration<double> grouping_time = (time[1] - time[0]);
		chrono::duration<double> ocr_time = (time[2] - time[1]);
		avg_time[4] += grouping_time.count();
		avg_time[5] += ocr_time.count();
		
		for (auto it : root_vec)
			er_filter->er_delete(it);
	}
	end = chrono::high_resolution_clock::now();
	avg_time[6] = chrono::duration<double>(end - start).count();
	avg_time[4] *= frame_count;
	avg_time[5] *= frame_count;

	//capture_thread.join();`
	cap.release();
	writer.release();

	std::cout << "Total frame number: " << img_count << "\n"
		<< "ER extraction = " << avg_time[0] * 1000 / img_count << "ms\n"
		<< "Non-maximum suppression = " << avg_time[1] * 1000 / img_count << "ms\n"
		<< "Classification = " << avg_time[2] * 1000 / img_count << "ms\n"
		<< "Character tracking = " << avg_time[3] * 1000 / img_count << "ms\n"
		<< "Character grouping = " << avg_time[4] * 1000 / img_count << "ms\n"
		<< "OCR = " << avg_time[5] * 1000 / img_count << "ms\n"
		<< "Total execution time = " << avg_time[6] * 1000 / img_count << "ms\n\n";

	fstream fout("video_result/result/time_log.txt", fstream::out);
	fout << "Total frame number: " << img_count << "\n"
		<< "ER extraction = " << avg_time[0] * 1000 / img_count << "ms\n"
		<< "Non-maximum suppression = " << avg_time[1] * 1000 / img_count << "ms\n"
		<< "Classification = " << avg_time[2] * 1000 / img_count << "ms\n"
		<< "Character tracking = " << avg_time[3] * 1000 / img_count << "ms\n"
		<< "Character grouping = " << avg_time[4] * 1000 / img_count << "ms\n"
		<< "OCR = " << avg_time[5] * 1000 / img_count << "ms\n"
		<< "Total execution time = " << avg_time[6] * 1000 / img_count << "ms\n\n";

	

#elif defined(IMAGE_MODE)
	int img_count = 0;
	vector<double> avg_time(7, 0);
	vector<vector<Text>> det_text;
	for (int n = 1; n <= 328; n++)
	{
		Mat src;
		Mat result;
		if (!load_challenge2_test_file(src, n))	continue;

		ERs root;
		vector<ERs> all;
		vector<ERs> pool;
		vector<ERs> strong;
		vector<ERs> weak;
		ERs tracked;
		vector<Text> result_text;

		vector<double> times = er_filter->text_detect(src, root, all, pool, strong, weak, tracked, result_text);
		show_result(src, result, text, times, tracked, strong, weak, all, pool);
		
		++img_count;
		for (int i = 0; i < times.size(); i++)
			avg_time[i] += times[i];
		det_text.push_back(text);
	}

	cout << "Total frame number: " << img_count << "\n"
		<< "ER extraction = " << avg_time[0] * 1000 / img_count << "ms\n"
		<< "Non-maximum suppression = " << avg_time[1] * 1000 / img_count << "ms\n"
		<< "Classification = " << avg_time[2] * 1000 / img_count << "ms\n"
		<< "Character tracking = " << avg_time[3] * 1000 / img_count << "ms\n"
		<< "Character grouping = " << avg_time[4] * 1000 / img_count << "ms\n"
		<< "OCR = " << avg_time[5] * 1000 / img_count << "ms\n"
		<< "Total execution time = " << avg_time[6] * 1000 / img_count << "ms\n\n";

	save_deteval_xml(det_text);
#endif

	delete er_filter->wtc;
	delete er_filter->stc;
	delete er_filter->ocr;
	return 0;
}