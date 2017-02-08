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


void save_biggest_er(string inImg, string outfile)
{
	ERFilter erFilter(1, 10, MAX_AREA, STABILITY_T, OVERLAP_COEF);
	
	Mat img = imread(inImg, IMREAD_UNCHANGED);
	if (img.empty())
		return;

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	
	ER *black_root = erFilter.er_tree_extract(gray);
	ER *white_root = erFilter.er_tree_extract(255-gray);
	ER *max_er = new ER();
	black_root->area = 0;
	white_root->area = 0;

	vector<ER *> tree_stack;
	ER *root = black_root;
save_step_21:
	for (; root != nullptr; root = root->child)
	{
		tree_stack.push_back(root);
		if (root->area > max_er->area && root->area < img.total())
		{
			max_er = root;
		}
	}

	if (root == nullptr && tree_stack.empty())
	{
		goto white;
	}

	root = tree_stack.back();
	tree_stack.pop_back();
	root = root->next;
	goto save_step_21;


white:
	tree_stack.clear();
	root = white_root;
save_step_22:
	for (; root != nullptr; root = root->child)
	{
		tree_stack.push_back(root);
		if (root->area > max_er->area && root->area < img.total())
		{
			max_er = root;
		}
	}

	if (root == nullptr && tree_stack.empty())
	{
		goto save;
	}

	root = tree_stack.back();
	tree_stack.pop_back();
	root = root->next;
	goto save_step_22;

save:
	imwrite(outfile, img(max_er->bound));
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
	double d1 = 0.60;
	double Ftarget2 = 0.10;
	double f2 = 0.80;
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


			cout << pic << "\tneg" << i << " finish " << endl;
		}
	}
	


	for (int i = 1; i <= 3; i++)
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


void save_pos_biggest_er()
{
	for (int i = 2; i <= 3; i++)
	{
		for (int pic = 1; pic <= 3500; pic++)
		{
			char filename[30];
			sprintf(filename, "res/pos%d/%d.jpg", i, pic);

			char outfile[30];
			sprintf(outfile, "res/tmp%d/%d.jpg", i, pic);
			
			save_biggest_er(filename, outfile);
		}
	}
}
