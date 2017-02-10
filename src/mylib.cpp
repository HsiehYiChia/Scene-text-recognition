#include "mylib.h"

#define THRESHOLD_STEP 2
#define MIN_AREA 50
#define MAX_AREA 90000
#define STABILITY_T 2
#define OVERLAP_COEF 0.7

#define MAX_WIDTH 800
#define MAX_HEIGHT 800


bool load_test_file(Mat &src, int n)
{
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

void calc_detection_rate(int n, vector<Text>)
{
	char filename[50];
	sprintf(filename, "res/ICDAR2015_test_GT/gt_img_%d.txt", n);
	fstream fin(filename, fstream::in);
	if (!fin.is_open())
	{
		std::cout << "Error: Ground Truth file " << n << " is not opened!!" << endl;
		return;
	}

	char picname[50];
	sprintf(picname, "res/ICDAR2015_test/img_%d.jpg", n);
	Mat src = imread(picname, CV_LOAD_IMAGE_UNCHANGED);


	vector<string> data;
	while (!fin.eof())
	{
		string buf;
		fin >> buf;
		data.push_back(buf);
	}

	// the last data would be eof, erase it
	data.pop_back();
	for (int i = 0; i < data.size(); i++)
	{
		data[i].pop_back();
		if (i % 5 == 4)
			data[i].erase(data[i].begin());
	}

	double resize_factor = 1.0;
	if (src.cols > MAX_WIDTH || src.rows > MAX_HEIGHT)
	{
		resize_factor = (src.rows > MAX_HEIGHT) ? (double)MAX_HEIGHT / src.rows : (double)MAX_WIDTH / src.cols;
	}

	// convert string numbers to bounding box, format as shown below
	// 0 0 100 100 HAHA
	// first 2 numbers represent the coordinate of top left point
	// last 2 numbers represent the coordinate of bottom right point
	vector<Rect> bbox;
	for (int i = 0; i < data.size(); i += 5)
	{
		int x1 = stoi(data[i]);
		int y1 = stoi(data[i + 1]);
		int x2 = stoi(data[i + 2]);
		int y2 = stoi(data[i + 3]);

		x1 *= resize_factor;
		y1 *= resize_factor;
		x2 *= resize_factor;
		y2 *= resize_factor;

		bbox.push_back(Rect(Point(x1, y1), Point(x2, y2)));
	}

	// merge the bounding box that could in the same group
	for (int i = 0; i < bbox.size(); i++)
	{
		for (int j = i+1; j < bbox.size(); j++)
		{
			if (abs(bbox[i].y - bbox[j].y) < bbox[i].height &&
				abs(bbox[i].y - bbox[j].y) < 0.2 * src.cols * resize_factor &&
				(double)min(bbox[i].height, bbox[j].height) / (double)max(bbox[i].height, bbox[j].height) > 0.7)
			{
				int x1 = min(bbox[i].x, bbox[j].x);
				int y1 = min(bbox[i].y, bbox[j].y);
				int x2 = max(bbox[i].br().x, bbox[j].br().x);
				int y2 = max(bbox[i].br().y, bbox[j].br().y);
				bbox[i] = Rect(Point(x1, y1), Point(x2, y2));
				bbox.erase(bbox.begin() + j);
				j--;
			}
		}
	}



	fin.close();
}


//==================================================
//=============== Training Function ================
//==================================================
void train_classifier()
{
	TrainingData *td1 = new TrainingData();
	TrainingData *td2 = new TrainingData();
	AdaBoost adb1(AdaBoost::REAL, AdaBoost::DECISION_STUMP, 30);
	AdaBoost adb2(AdaBoost::REAL, AdaBoost::DECISION_STUMP, 60);


	td1->read_data("er_classifier/training_data.txt");
	adb1.train_classifier(*td1, "er_classifier/adb1.classifier");
	
	for (int i = 0; i < td1->data.size(); i++)
	{
		if (adb1.predict(td1->data[i].fv) < 2.0)
		{
			td2->data.push_back(td1->data[i]);
		}
	}
	
	delete td1;


	td2->set_num(td2->data.size());
	td2->set_dim(td2->data.front().fv.size());
	adb2.train_classifier(*td2, "er_classifier/adb2.classifier");
}


void train_cascade()
{
	double Ftarget1 = 0.02;
	double f1 = 0.80;
	double d1 = 0.80;
	double Ftarget2 = 0.30;
	double f2 = 0.90;
	double d2 = 0.90;
	TrainingData *td1 = new TrainingData();
	TrainingData *tmp = new TrainingData();
	TrainingData *td2 = new TrainingData();
	AdaBoost *adb1 = new CascadeBoost(AdaBoost::REAL, AdaBoost::DECISION_STUMP, Ftarget1, f1, d1);
	AdaBoost *adb2 = new CascadeBoost(AdaBoost::REAL, AdaBoost::DECISION_STUMP, Ftarget2, f2, d2);

	freopen("er_classifier/log.txt", "w", stdout);

	cout << "Strong Text    Ftarget:" << Ftarget1 << " f=" << f1 << " d:" << d1 << endl;
	td1->read_data("er_classifier/training_data.txt");
	adb1->train_classifier(*td1, "er_classifier/cascade1.classifier");

	cout << endl << "Weak Text    Ftarget:" << Ftarget2 << " f=" << f2 << " d:" << d2 << endl;
	td2->read_data("er_classifier/training_data.txt");
	adb2->train_classifier(*td2, "er_classifier/cascade2.classifier");
}


void get_canny_data()
{
	fstream fout = fstream("er_classifier/training_data.txt", fstream::out);

	ERFilter erFilter(THRESHOLD_STEP, MIN_AREA, MAX_AREA, STABILITY_T, OVERLAP_COEF);
	erFilter.ocr = new OCR();

	const int N = 2;
	const int normalize_size = 24;

	for (int i = 1; i <= 4; i++)
	{
		for (int pic = 1; pic <= 10000; pic++)
		{
			char filename[30];
			sprintf(filename, "res/neg%d/%d.jpg", i, pic);

			Mat input = imread(filename, IMREAD_GRAYSCALE);
			if (input.empty())	continue;

			vector<double> spacial_hist = erFilter.make_LBP_hist(input, N, normalize_size);
			fout << -1;
			for (int f = 0; f < spacial_hist.size(); f++)
				fout << " " << spacial_hist[f];
			fout << endl;

			spacial_hist = erFilter.make_LBP_hist(255-input, N, normalize_size);
			fout << -1;
			for (int f = 0; f < spacial_hist.size(); f++)
				fout << " " << spacial_hist[f];
			fout << endl;


			cout << pic << "\tneg" << i << " finish " << endl;
		}
	}
	


	for (int i = 1; i <= 4; i++)
	{
		for (int pic = 1; pic <= 4000; pic++)
		{
			char filename[30];
			sprintf(filename, "res/pos%d/%d.jpg", i, pic);

			Mat input = imread(filename, IMREAD_GRAYSCALE);
			if (input.empty())	continue;

			vector<double> spacial_hist = erFilter.make_LBP_hist(input, N, normalize_size);
			fout << 1;
			for (int f = 0; f < spacial_hist.size(); f++)
				fout << " " << spacial_hist[f];
			fout << endl;

			spacial_hist = erFilter.make_LBP_hist(255 - input, N, normalize_size);
			fout << 1;
			for (int f = 0; f < spacial_hist.size(); f++)
				fout << " " << spacial_hist[f];
			fout << endl;

			cout << pic << "\tpos" << i <<" finish " << endl;
		}
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


void opencv_train()
{
	using namespace ml;
	Ptr<Boost> boost = Boost::create();
	Ptr<TrainData> trainData = TrainData::loadFromCSV("er_classifier/training_data.txt", 0, 0, 1, String(), ' ');
	boost->setBoostType(Boost::REAL);
	boost->setWeakCount(100);
	boost->setMaxDepth(1);
	boost->setWeightTrimRate(0);
	cout << "training..." << endl;
	boost->train(trainData);
	boost->save("er_classifier/opencv_classifier.xml");
}