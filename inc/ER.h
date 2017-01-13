#ifndef __ERTREMAL_REGION__
#define __ERTREMAL_REGION__

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <memory>

#include <opencv.hpp>
#include "adaboost.h"
#include "OCR.h"


using namespace std;
using namespace cv;


struct ER
{
public:
	//! Constructor
	ER() {};
	ER(const int level_, const int pixel_, const int x_, const int y_);

	//! seed point and threshold (max grey-level value)
	int pixel;
	int level;
	int area;
	Rect bound;                //!< bounding box
	bool done;
	double stability;

	//! pointers preserving the tree structure of the component tree
	ER* parent;
	ER* child;
	ER* next;

	//! for OCR
	char c;
	char prob;
};

typedef vector<ER *> ERs;

class ERFilter
{
public:
	ERFilter(int thresh_step = 2, int min_area = 100, int max_area = 100000, int stability_t = 2, double overlap_coef = 0.7);
	~ERFilter()	{}
	
	//! modules
	AdaBoost *adb;
	OCR *ocr;

	//! functions
	ER* er_tree_extract(Mat input);
	void non_maximum_supression(ER *er, ERs &pool, Mat input);
	void classify(ERs pool, Mat input);
	vector<double> make_LBP_hist(Mat input, const int N = 2, const int normalize_size = 24);

private:
	//! Parameters
	const int THRESH_STEP;
	const int MIN_AREA;
	const int MAX_AREA;
	const int STABILITY_T;
	const double OVERLAP_COEF;
	enum { right, bottom, left, top };

	//! ER operation functions
	inline void er_accumulate(ER *er, const int &current_pixel, const int &x, const int &y);
	void er_merge(ER *parent, ER *child);
	void er_save(ER *er);
	void process_stack(const int new_pixel_grey_level, vector<ER *> &er_stack);

	// feature extract
	
	Mat calc_LBP(Mat input, const int size = 24);
};
#endif