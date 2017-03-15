#include "utils.h"


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


void show_result(Mat &src, vector<ERs> &all, vector<ERs> &pool, vector<ERs> &strong, vector<ERs> &weak, ERs &tracked, vector<Text> &text)
{
	Mat all_img = src.clone();
	Mat pool_img = src.clone();
	Mat strong_img = src.clone();
	Mat weak_img = src.clone();
	Mat tracked_img = src.clone();
	Mat result_img = src.clone();
	for (int i = 0; i < pool.size(); i++)
	{
		for (auto it : all[i])
			rectangle(all_img, it->bound, Scalar(255, 0, 0));

		for (auto it : pool[i])
			rectangle(pool_img, it->bound, Scalar(255, 0, 0));

		for (auto it : weak[i])
			rectangle(weak_img, it->bound, Scalar(0, 0, 255));

		for (auto it : strong[i])
			rectangle(strong_img, it->bound, Scalar(0, 255, 0));
	}

	for (auto it : tracked)
	{
		rectangle(tracked_img, it->bound, Scalar(255, 0, 255));
	}

	for (auto it : text)
	{
		rectangle(result_img, it.box, Scalar(0, 255, 255), 2);
	}

#ifdef DO_OCR
	for (int i = 0; i < text.size(); i++)
	{
		putText(result_img, text[i].word, text[i].box.tl(), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 2);
	}
#endif

	double alpha = 0.7;
	addWeighted(all_img, alpha, src, 1.0 - alpha, 0.0, all_img);
	cv::imshow("all", all_img);
	cv::imshow("pool", pool_img);
	cv::imshow("weak", weak_img);
	cv::imshow("strong", strong_img);
	cv::imshow("tracked", tracked_img);
	cv::imshow("result", result_img);

#ifndef WEBCAM_MODE
	imwrite("all.jpg", all_img);
	imwrite("pool.jpg", pool_img);
	imwrite("weak.jpg", weak_img);
	imwrite("strong.jpg", strong_img);
	imwrite("tracked.jpg", tracked_img);
	imwrite("result.jpg", result_img);
	waitKey(0);
#endif

}



void draw_linear_time_MSER(string img_name)
{
	Mat input = imread(img_name);

	int pixel_count = 0;
	VideoWriter writer;
	//writer.open("Linear time MSER.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, input.size());
	writer.open("Linear time MSER.wmv", CV_FOURCC('W', 'M', 'V', '2'), 30, input.size());


	Mat color = Mat::zeros(input.rows, input.cols, CV_8UC3);
	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);
	const int width = gray.cols;
	const int height = gray.rows;
	const int highest_level = 255 + 1;
	const uchar *imgData = gray.data;

	//!< 1. Clear the accessible pixel mask, the heap of boundary pixels and the component
	bool *pixel_accessible = new bool[height*width]();
	vector<int> boundary_pixel[256];
	vector<int> boundary_edge[256];
	vector<ER *>er_stack;

	int priority = highest_level;


	//!< 1-2. push a dummy-component onto the stack, 
	//!<	  with grey-level heigher than any allowed in the image
	er_stack.push_back(new ER(256, 0, 0, 0));


	//!< 2. make the top-right corner the source pixel, get its gray level and mark it accessible
	int current_pixel = 0;
	int current_edge = 0;
	int current_level = imgData[current_pixel];
	pixel_accessible[current_pixel] = true;


step_3:
	int x = current_pixel % width;
	int y = current_pixel / width;

	//!< 3. push an empty component with current_level onto the component stack
	er_stack.push_back(new ER(current_level, current_pixel, x, y));


	for (;;)
	{
		//!< 4. Explore the remaining edges to the neighbors of the current pixel, in order, as follows : 
		//!<	For each neighbor, check if the neighbor is already accessible.If it
		//!<	is not, mark it as accessible and retrieve its grey - level.If the grey - level is not
		//!<	lower than the current one, push it onto the heap of boundary pixels.If on
		//!<	the other hand the grey - level is lower than the current one, enter the current
		//!<	pixel back into the queue of boundary pixels for later processing(with the
		//!<	next edge number), consider the new pixel and its grey - level and go to 3.
		int neighbor_pixel;
		int neighbor_level;


		for (; current_edge < 4; current_edge++)
		{
			switch (current_edge)
			{
			case 0: neighbor_pixel = (x + 1 < width) ? current_pixel + 1 : current_pixel;	break;
			case 1: neighbor_pixel = (y + 1 < height) ? current_pixel + width : current_pixel;	break;
			case 2: neighbor_pixel = (x > 0) ? current_pixel - 1 : current_pixel;	break;
			case 3: neighbor_pixel = (y > 0) ? current_pixel - width : current_pixel;	break;
			default: break;
			}

			if (!pixel_accessible[neighbor_pixel] && neighbor_pixel != current_pixel)
			{
				pixel_accessible[neighbor_pixel] = true;
				neighbor_level = imgData[neighbor_pixel];

				if (neighbor_level >= current_level)
				{
					boundary_pixel[neighbor_level].push_back(neighbor_pixel);
					boundary_edge[neighbor_level].push_back(0);

					if (neighbor_level < priority)
						priority = neighbor_level;

					int nx = neighbor_pixel % width;
					int ny = neighbor_pixel / width;
					color.at<uchar>(ny, nx * 3) = 0;
					color.at<uchar>(ny, nx * 3 + 1) = 255;
					color.at<uchar>(ny, nx * 3 + 2) = 0;
				}
				else
				{
					boundary_pixel[current_level].push_back(current_pixel);
					boundary_edge[current_level].push_back(current_edge + 1);

					if (current_level < priority)
						priority = current_level;

					color.at<uchar>(y, x * 3) = 0;
					color.at<uchar>(y, x * 3 + 1) = 255;
					color.at<uchar>(y, x * 3 + 2) = 0;

					current_pixel = neighbor_pixel;
					current_level = neighbor_level;
					current_edge = 0;
					goto step_3;
				}
			}
		}




		//!< 5. Accumulate the current pixel to the component at the top of the stack 
		//!<	(water saturates the current pixel).
		er_stack.back()->area++;
		int x1 = min(er_stack.back()->bound.x, x);
		int x2 = max(er_stack.back()->bound.br().x - 1, x);
		int y1 = min(er_stack.back()->bound.y, y);
		int y2 = max(er_stack.back()->bound.br().y - 1, y);
		er_stack.back()->bound.x = x1;
		er_stack.back()->bound.y = y1;
		er_stack.back()->bound.width = x2 - x1 + 1;
		er_stack.back()->bound.height = y2 - y1 + 1;

		color.at<uchar>(y, x * 3) = input.at<uchar>(y, x * 3);
		color.at<uchar>(y, x * 3 + 1) = input.at<uchar>(y, x * 3 + 1);
		color.at<uchar>(y, x * 3 + 2) = input.at<uchar>(y, x * 3 + 2);
		/*color.at<uchar>(y, x * 3) = current_level;
		color.at<uchar>(y, x * 3 + 1) = current_level;
		color.at<uchar>(y, x * 3 + 2) = current_level;*/
		pixel_count++;
		if (pixel_count % 300 == 0)
		{
			imshow("Linear time MSER", color);
			writer << color;
			waitKey(1);
		}


		//!< 6. Pop the heap of boundary pixels. If the heap is empty, we are done. If the
		//!<	returned pixel is at the same grey - level as the previous, go to 4	
		if (priority == highest_level)
		{
			delete[] pixel_accessible;
			writer.release();
			waitKey(0);
			return;
		}


		int new_pixel = boundary_pixel[priority].back();
		int new_edge = boundary_edge[priority].back();
		int new_pixel_grey_level = imgData[new_pixel];

		boundary_pixel[priority].pop_back();
		boundary_edge[priority].pop_back();

		while (boundary_pixel[priority].empty() && priority < highest_level)
			priority++;

		current_pixel = new_pixel;
		current_edge = new_edge;
		x = current_pixel % width;
		y = current_pixel / width;

		if (new_pixel_grey_level != current_level)
		{
			
			current_level = new_pixel_grey_level;

			do
			{
				ER *top = er_stack.back();
				ER *second_top = er_stack.end()[-2];
				er_stack.pop_back();

				
				if (new_pixel_grey_level < second_top->level)
				{
					er_stack.push_back(new ER(new_pixel_grey_level, top->pixel, top->x, top->y));
					er_stack.back()->area += top->area;

					const int x1 = min(er_stack.back()->bound.x, top->bound.x);
					const int x2 = max(er_stack.back()->bound.br().x - 1, top->bound.br().x - 1);
					const int y1 = min(er_stack.back()->bound.y, top->bound.y);
					const int y2 = max(er_stack.back()->bound.br().y - 1, top->bound.br().y - 1);

					er_stack.back()->bound.x = x1;
					er_stack.back()->bound.y = y1;
					er_stack.back()->bound.width = x2 - x1 + 1;
					er_stack.back()->bound.height = y2 - y1 + 1;

					
					top->next = er_stack.back()->child;
					er_stack.back()->child = top;
					top->parent = er_stack.back();

					break;
				}

				second_top->area += top->area;

				const int x1 = min(second_top->bound.x, top->bound.x);
				const int x2 = max(second_top->bound.br().x - 1, top->bound.br().x - 1);
				const int y1 = min(second_top->bound.y, top->bound.y);
				const int y2 = max(second_top->bound.br().y - 1, top->bound.br().y - 1);

				second_top->bound.x = x1;
				second_top->bound.y = y1;
				second_top->bound.width = x2 - x1 + 1;
				second_top->bound.height = y2 - y1 + 1;


				top->next = second_top->child;
				second_top->child = top;
				top->parent = second_top;

			}
			while (new_pixel_grey_level > er_stack.back()->level);
		}
	}
}


void draw_multiple_channel(string img_name)
{
	ERFilter *erFilter = new ERFilter(1, MIN_ER_AREA, MAX_ER_AREA, NMS_STABILITY_T, NMS_OVERLAP_COEF);
	Mat input = imread(img_name);
	


	Mat Ycrcb;
	vector<Mat> channel;
	erFilter->compute_channels(input, Ycrcb, channel);

	ERs root;
	vector<ERs> all;
	vector<ERs> pool;
	root.resize(channel.size());
	all.resize(channel.size());
	pool.resize(channel.size());

	for (int i = 0; i < channel.size(); i++)
	{
		root[i] = erFilter->er_tree_extract(channel[i]);
		erFilter->non_maximum_supression(root[i], all[i], pool[i], channel[i]);

		for (int j = 0; j < pool[i].size(); j++)
		{
			rectangle(channel[i%3], pool[i][j]->bound, Scalar(0));
		}
	}


	imshow("Y", channel[0]);
	imshow("Cr", channel[1]);
	imshow("Cb", channel[2]);
	imwrite("Y.jpg", channel[0]);
	imwrite("Cr.jpg", channel[1]);
	imwrite("Cb.jpg", channel[2]);
	waitKey(0);
}


void test_MSER_time(string img_name)
{
	fstream fout = fstream("time.txt", fstream::out);
	ERFilter *erFilter = new ERFilter(1, 100, 1.0E7, NMS_STABILITY_T, NMS_OVERLAP_COEF);
	Mat input = imread(img_name, IMREAD_GRAYSCALE);
	double coef = 0.2;
	Mat tmp;
	while (tmp.total() <= 1.0E8)
	{
		resize(input, tmp, Size(), coef, coef);

		chrono::high_resolution_clock::time_point start, end;
		start = chrono::high_resolution_clock::now();

		ER *root = erFilter->er_tree_extract(tmp);

		end = chrono::high_resolution_clock::now();

		erFilter->er_delete(root);
		std::cout << "pixel number: " << tmp.total();
		std::cout << "\ttime: " << chrono::duration<double>(end - start).count() * 1000 << "ms\n";
		fout << tmp.total() << " " << chrono::duration<double>(end - start).count() * 1000 << endl;
		coef += 0.1;
	}
	erFilter->er_tree_extract(input);
}


vector<Vec4i> load_gt(int n)
{
	char filename[50];
	sprintf(filename, "res/ICDAR2015_test_GT/gt_img_%d.txt", n);
	fstream fin(filename, fstream::in);
	if (!fin.is_open())
	{
		std::cout << "Error: Ground Truth file " << n << " is not opened!!" << endl;
		return vector<Vec4i>();
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
	/*for (int i = 0; i < bbox.size(); i++)
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
	}*/

	vector<Vec4i> gt;
	for (int i = 0; i < bbox.size(); i++)
	{
		gt.push_back(Vec4i(bbox[i].x, bbox[i].y, bbox[i].width, bbox[i].height));
	}

	fin.close();
	return gt;
}


Vec6d calc_detection_rate(int n, vector<Text> &text)
{
	vector<Vec4i> gt = load_gt(n);
	vector<Rect> gt_box;
	for (int i = 0; i < gt.size(); i++)
	{
		gt_box.push_back(Rect(gt[i][0], gt[i][1], gt[i][2], gt[i][3]));
	}


	vector<bool> detected(gt_box.size());
	double tp = 0;
	for (int i = 0; i < gt_box.size(); i++)
	{
		for (int j = 0; j < text.size(); j++)
		{
			Rect overlap = gt_box[i] & text[j].box;
			Rect union_box = gt_box[i] | text[j].box;

			if ((double)overlap.area() / union_box.area() > 0.3)
			{
				detected[i] = true;
				tp++;
				break;
			}
		}
	}

	double recall = tp / detected.size();
	double precision = (!text.empty()) ? tp / text.size() : 0;
	double harmonic_mean = (precision != 0) ? (recall*precision) * 2 / (recall + precision) : 0;

	return Vec6d(tp, detected.size(), text.size(), recall, precision, harmonic_mean);
}



void save_deteval_xml(vector<vector<Text>> &text)
{
	remove("gt.xml");
	remove("det.xml");
	fstream fgt("gt.xml", fstream::out);
	fstream fdet("det.xml", fstream::out);

	fgt << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << endl << "<tagset>" << endl;
	fdet << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << endl << "<tagset>" << endl;

	for (int i = 1; i <= 233; i++)
	{
		vector<Vec4i> gt = load_gt(i);
		if (gt.empty()) continue;

		// ground truth
		fgt << "\t<image>" << endl << "\t\t" << "<imageName>img_" << i << ".jpg</imageName>" << endl;
		fgt << "\t\t<taggedRectangles>" << endl;
		for (int j = 0; j < gt.size(); j++)
		{
			fgt << "\t\t\t<taggedRectangle x=\"" << gt[j][0] << "\" y=\"" << gt[j][1] << "\" width=\"" << gt[j][2] << "\" height=\"" << gt[j][3] << "\" offset=\"0\" />" << endl;
		}
		fgt << "\t\t</taggedRectangles>" << endl;
		fgt << "\t</image>" << endl;


		// detections
		fdet << "\t<image>" << endl << "\t\t" << "<imageName>img_" << i << ".jpg</imageName>" << endl;
		fdet << "\t\t<taggedRectangles>" << endl;
		for (int j = 0; j < text[i-1].size(); j++)
		{
			fdet << "\t\t\t<taggedRectangle x=\"" << text[i-1][j].box.x << "\" y=\"" << text[i - 1][j].box.y << "\" width=\"" << text[i - 1][j].box.width << "\" height=\"" << text[i - 1][j].box.height << "\" offset=\"0\" />" << endl;
		}
		fdet << "\t\t</taggedRectangles>" << endl;
		fdet << "\t</image>" << endl;
	}

	fgt << "</tagset>";
	fdet << "</tagset>";
}


void draw_all_er()
{
	Mat haha= imread("haha.jpg");

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


void bootstrap()
{
	ERFilter *erFilter = new ERFilter(THRESHOLD_STEP, MIN_ER_AREA, MAX_ER_AREA, NMS_STABILITY_T, NMS_OVERLAP_COEF);
	erFilter->adb1 = new CascadeBoost("er_classifier/cascade1.classifier");
	erFilter->adb2 = new CascadeBoost("er_classifier/cascade2.classifier");


	int i = 0;
	for (int pic = 1; pic <= 10000; pic++)
	{
		char filename[100];
		//sprintf(filename, "res/neg/image_net_neg/%d.jpg", pic);
		sprintf(filename, "D:\\0.Projects\\image_data_set\\ICDAR2015\\Challenge1\\Challenge1_Test_Task3_Images\\word_%d.png", pic);
	

		ERs all, pool, strong, weak;
		Mat input = imread(filename, IMREAD_GRAYSCALE);
		if (input.empty()) continue;


		ER *root = erFilter->er_tree_extract(input);
		erFilter->non_maximum_supression(root, all, pool, input);
		erFilter->classify(pool, strong, weak, input);


		for (auto it : strong)
		{
			char imgname[30];
			sprintf(imgname, "res/tmp1/%d_%d.jpg", pic, i);
			imwrite(imgname, input(it->bound));
			i++;
		}

		for (auto it : weak)
		{
			char imgname[30];
			sprintf(imgname, "res/tmp1/%d_%d.jpg", pic, i);
			imwrite(imgname, input(it->bound));
			i++;
		}

		cout << pic << " finish " << endl;
	}
}



void get_canny_data()
{
	fstream fout = fstream("er_classifier/training_data.txt", fstream::out);

	ERFilter erFilter(THRESHOLD_STEP, MIN_ER_AREA, MAX_ER_AREA, NMS_STABILITY_T, NMS_OVERLAP_COEF);
	erFilter.ocr = new OCR();

	const int N = 2;
	const int normalize_size = 24;

	for (int i = 1; i <= 3; i++)
	{
		for (int pic = 0; pic <= 15000; pic++)
		{
			char filename[30];
			sprintf(filename, "res/neg/neg%d/%d.jpg", i, pic);

			Mat input = imread(filename, IMREAD_GRAYSCALE);
			if (input.empty())	continue;

			vector<double> spacial_hist = erFilter.make_LBP_hist(input, N, normalize_size);
			fout << -1;
			for (int f = 0; f < spacial_hist.size(); f++)
			{
				fout << " " << spacial_hist[f];
			}
			fout << endl;

			/*spacial_hist = erFilter.make_LBP_hist(255-input, N, normalize_size);
			fout << -1;
			for (int f = 0; f < spacial_hist.size(); f++)
				fout << " " << spacial_hist[f];
			fout << endl;*/


			cout << filename << " finish " << endl;
		}
	}
	


	for (int i = 1; i <= 3; i++)
	{
		for (int pic = 0; pic <= 15000; pic++)
		{
			char filename[30];
			sprintf(filename, "res/pos/pos%d/%d.jpg", i, pic);

			Mat input = imread(filename, IMREAD_GRAYSCALE);
			if (input.empty())	continue;

			vector<double> spacial_hist = erFilter.make_LBP_hist(input, N, normalize_size);
			fout << 1;
			for (int f = 0; f < spacial_hist.size(); f++)
			{
				fout << " " << spacial_hist[f];
			}
			fout << endl;

			spacial_hist = erFilter.make_LBP_hist(255 - input, N, normalize_size);
			fout << 1;
			for (int f = 0; f < spacial_hist.size(); f++)
			{
				fout << " " << spacial_hist[f];
			}
			fout << endl;

			cout << filename <<" finish " << endl;
		}
	}
	
}


void rotate_image()
{
	vector<string> font_name = { "Cambria", "Coda", "Comic_Sans_MS", "Courier_New", "Domine", "Droid_Serif", "Fine_Ming", "Francois_One", "Georgia", "Impact",
		"Neuton", "Play", "PT_Serif", "Russo_One", "Sans_Serif", "Syncopate", "Time_New_Roman", "Trebuchet_MS", "Twentieth_Century", "Verdana" };
	vector<string> font_type = { "Bold", "Bold_and_Italic", "Italic", "Normal" };
	vector<string> cat = { "lower", "upper", "number" };

	OCR ocr = OCR();
	int n = 0;
	double rad = 0/180.0*CV_PI;
	for (int i = 0; i < font_name.size(); i++)
	{
		for (int j = 0; j < font_type.size(); j++)
		{
			for (int k = 0; k < 3; k++)
			{
				string path = String("ocr_classifier/" + font_name[i] + "/" + font_type[j] + "/" + cat[k] + "/");
				string c;
				int loop_num;
				if (k == 0 || k == 1)
				{
					c = { 'a', '.', 'j', 'p', 'g', '\0' };
					loop_num = 26;
				}
				else if (k == 2)
				{
					c = { '0', '.', 'j', 'p', 'g', '\0' };
					loop_num = 10;
				}
				for (int i = 0; i < loop_num; i++)
				{
					String filename = path + c;
					cout << filename << endl;

					Mat img = imread(filename, IMREAD_GRAYSCALE);
					c[0]++;

					if (img.empty()) continue;

					ocr.geometric_normalization(img, img, rad, false);

					char buf[256];
					sprintf(buf, "res/tmp1/%d.jpg",  n++);
					imwrite(buf, img);
				}
			}
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


	ERFilter erFilter(THRESHOLD_STEP, MIN_ER_AREA, MAX_ER_AREA, NMS_STABILITY_T, NMS_OVERLAP_COEF);
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