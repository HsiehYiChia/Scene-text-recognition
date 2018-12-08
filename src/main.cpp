#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>

#include "../inc/ER.h"
#include "../inc/OCR.h"
#include "../inc/adaboost.h"
#include "../inc/utils.h"

#ifdef _WIN32
#include "../inc/getopt.h"
#elif __linux__
#include <getopt.h>
#endif


using namespace std;

int main(int argc, char* argv[])
{
	ERFilter* er_filter = new ERFilter(THRESHOLD_STEP, MIN_ER_AREA, MAX_ER_AREA, NMS_STABILITY_T, NMS_OVERLAP_COEF, MIN_OCR_PROBABILITY);
	er_filter->stc = new CascadeBoost("classifier/strong.classifier");
	er_filter->wtc = new CascadeBoost("classifier/weak.classifier");
	er_filter->ocr = new OCR("classifier/OCR.model", OCR_IMG_L, OCR_FEATURE_L);
	er_filter->load_tp_table("dictionary/tp_table.txt");
	er_filter->corrector.load("dictionary/big.txt");

	char *filename = nullptr;
	char *training_type = nullptr;
	int is_file = -1;
	char c = 0;
	
	while ((c = getopt (argc, argv, "v:i:o:l:t:")) != -1)
	{
		switch (c)
		{
		case 'i':
			filename = optarg;
			image_mode(er_filter, filename);
			break;
		case 'v':
			filename = optarg;
			video_mode(er_filter, filename);
			break;
		case 'l':
			filename = optarg;
			draw_linear_time_MSER(filename);
			break;
		case 't':
			training_type = optarg;
			if (strcmp(training_type, "detection")==0)
			{
				get_lbp_data();
				train_detection_classifier();
			}
			else if (strcmp(training_type, "ocr") == 0)
			{
				get_ocr_data();
				train_ocr_model();
			}
		case '?':
			/* Camera Mode */
			if (optopt == 'v' && isprint(optopt))
				video_mode(er_filter, nullptr);
			else if (optopt == 'i' || optopt =='l')
			{
				cout << "Option -" << (char)optopt << " requires an argument" << endl;
				usage();
			}
			break;
		default:
			usage();
			abort();
		}
	}


	delete er_filter->wtc;
	delete er_filter->stc;
	delete er_filter->ocr;
	return 0;
}
