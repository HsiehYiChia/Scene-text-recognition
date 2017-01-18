#include "mylib.h"

#define THRESHOLD_STEP 2
#define MIN_AREA 50
#define MAX_AREA 90000
#define STABILITY_T 2
#define OVERLAP_COEF 0.7



bool load_test_file(Mat &src, int n)
{
	const int MAX_WIDTH = 800;
	const int MAX_HEIGHT = 800;

	char filename[50];
	sprintf(filename, "res/ICDAR2015_test/img_%d.jpg", n);
	src = imread(filename, CV_LOAD_IMAGE_UNCHANGED);

	if (src.empty())
	{
		std::cout << n << "\tFail to open" << filename << endl;
		return false;
	}

	else if (src.cols > MAX_WIDTH || src.rows > MAX_HEIGHT)
	{
		std::cout << n << "\t" << src.rows << "," << src.cols << "\tResize the image" << endl;
		double resize_factor = (src.rows > MAX_HEIGHT) ? (double)MAX_HEIGHT / src.rows : (double)MAX_WIDTH / src.cols;

		resize(src, src, Size(src.cols*resize_factor, src.rows*resize_factor));
		return true;
	}

	std::cout << n << "\t" << src.rows << "," << src.cols << endl;
	return true;
}


void compute_channels(Mat &src, Mat &YCrcb, vector<Mat> &channels)
{
	vector<Mat> splited;

	cv::cvtColor(src, YCrcb, COLOR_BGR2YCrCb);
	split(YCrcb, splited);

	channels.push_back(splited[0]);
	channels.push_back(splited[1]);
	channels.push_back(splited[2]);
	channels.push_back(255 - splited[0]);
	channels.push_back(255 - splited[1]);
	channels.push_back(255 - splited[2]);
}



// for training
void train_classifier()
{
	TrainingData td;
	td.read_data("er_classifier/training_data.txt");
	AdaBoost adb(AdaBoost::REAL, AdaBoost::DECISION_STUMP, 100);
	adb.train_classifier(td, "er_classifier/lbp.classifier");
}



void get_canny_data()
{
	fstream fout = fstream("er_classifier/training_data.txt", fstream::out);

	ERFilter erFilter(THRESHOLD_STEP, MIN_AREA, MAX_AREA, STABILITY_T, OVERLAP_COEF);
	erFilter.ocr = new OCR();

	const int N = 2;
	const int normalize_size = 24;

	for (int pic = 1; pic <= 8489; pic++)
	{
		char filename[30];
		sprintf(filename, "res/neg/neg (%d).jpg", pic);

		Mat input = imread(filename, IMREAD_GRAYSCALE);

		vector<double> spacial_hist = erFilter.make_LBP_hist(input, 2, 24);


		fout << -1;
		for (int f = 0; f < spacial_hist.size(); f++)
			fout << " " << spacial_hist[f];
		fout << endl;


		cout << pic << "neg finish" << endl;
	}

	for (int pic = 1; pic <= 6789; pic++)
	{
		char filename[30];
		sprintf(filename, "res/neg2/%d.jpg", pic);

		Mat input = imread(filename, IMREAD_GRAYSCALE);

		vector<double> spacial_hist = erFilter.make_LBP_hist(input, 2, 24);


		fout << -1;
		for (int f = 0; f < spacial_hist.size(); f++)
			fout << " " << spacial_hist[f];
		fout << endl;


		cout << pic << "neg2 finish" << endl;
	}

	for (int pic = 1; pic <= 3439; pic++)
	{
		char filename[30];
		sprintf(filename, "res/pos/pos (%d).jpg", pic);

		Mat input = imread(filename, IMREAD_GRAYSCALE);

		vector<double> spacial_hist = erFilter.make_LBP_hist(input, N, normalize_size);

		fout << 1;
		for (int f = 0; f < spacial_hist.size(); f++)
			fout << " " << spacial_hist[f];
		fout << endl;


		cout << pic << "pos finish" << endl;
	}

	for (int pic = 1; pic <= 6185; pic++)
	{
		char filename[30];
		sprintf(filename, "res/pos2/%d.jpg", pic);

		Mat input = imread(filename, IMREAD_GRAYSCALE);

		vector<double> spacial_hist = erFilter.make_LBP_hist(input, N, normalize_size);

		fout << 1;
		for (int f = 0; f < spacial_hist.size(); f++)
			fout << " " << spacial_hist[f];
		fout << endl;


		cout << pic << "pos2 finish" << endl;
	}

	for (int pic = 1; pic <= 5430; pic++)
	{
		char filename[30];
		sprintf(filename, "res/pos3/%d.jpg", pic);

		Mat input = imread(filename, IMREAD_GRAYSCALE);

		vector<double> spacial_hist = erFilter.make_LBP_hist(input, N, normalize_size);

		fout << 1;
		for (int f = 0; f < spacial_hist.size(); f++)
			fout << " " << spacial_hist[f];
		fout << endl;


		cout << pic << "pos3 finish" << endl;
	}
}


void get_ocr_data(int argc, char **argv, int type)
{
	char *in_img = nullptr;
	char *outfile = nullptr;
	int label = 0;
	if (argc != 4)
	{
		cout << "wrong input format" << endl;
		return;
	}

	else
	{
		in_img = argv[1];
		outfile = argv[2];
		label = atoi(argv[3]);
	}



	Mat input = imread(in_img, IMREAD_GRAYSCALE);
	if (input.empty())
	{
		cout << "No such file:" << in_img << endl;
		return;
	}


	ERFilter erFilter(THRESHOLD_STEP, MIN_AREA, MAX_AREA, STABILITY_T, OVERLAP_COEF);
	erFilter.ocr = new OCR();

	fstream fout = fstream(outfile, fstream::app);
	fout << label;

	if (type == 0)
	{
		Mat ocr_img;
		threshold(255 - input, ocr_img, 128, 255, CV_THRESH_OTSU);
		erFilter.ocr->rotate_mat(ocr_img, ocr_img, 0, true);
		erFilter.ocr->ARAN(ocr_img, ocr_img, 35);

		svm_node *fv = new svm_node[201];
		erFilter.ocr->extract_feature(ocr_img, fv);

		int i = 0;
		while (fv[i].index != -1)
		{
			fout << " " << fv[i].index << ":" << fv[i].value;
			i++;
		}

		fout << endl;
	}

	else if (type == 1)
	{
		const int N = 2;
		const int normalize_size = 24;

		vector<double> spacial_hist = erFilter.make_LBP_hist(input, N, normalize_size);

		double scale = (normalize_size / 2) * (normalize_size / 2);
		for (int f = 0; f < spacial_hist.size(); f++)
		{
			if (spacial_hist[f] != 0)
				fout << " " << f << ":" << spacial_hist[f] / 129.0;
		}

		fout << endl;
	}

	return;
}