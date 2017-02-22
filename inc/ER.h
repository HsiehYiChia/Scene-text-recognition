#ifndef __ERTREMAL_REGION__
#define __ERTREMAL_REGION__

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <forward_list>
#include <chrono>
#include <omp.h>
#include <memory>

#include <opencv.hpp>
#include "adaboost.h"
#include "OCR.h"

#define DO_OCR

using namespace std;
using namespace cv;

struct plist 
{
	plist() :p(0), next(nullptr) {};
	plist(int _p) :p(_p), next(nullptr) {};
	plist(plist *_next) :p(0), next(_next) {};
	plist(int _p, plist *_next) :p(_p), next(_next) {};

	int p;
	plist *next;
};


struct ER
{
public:
	//! Constructor
	ER() {};
	ER(const int level_, const int pixel_, const int x_, const int y_);

	//! seed point and threshold (max grey-level value)
	int pixel;
	int level;
	int x;
	int y;
	
	//! feature
	int area;
	Rect bound;
	Point center;
	double color1;
	double color2;
	double color3;
	double stkw;

	// for non maximal supression
	bool done;
	double stability;

	//! pointers preserving the tree structure of the component tree
	ER* parent;
	ER* child;
	ER* next;
	ER *sibling_L;
	ER *sibling_R;

	//! for OCR
	int ch;
	char letter;
	double prob;
};

typedef vector<ER *> ERs;

struct Text
{
	Text(){};
	Text(ER *x, ER *y, ER *z)
	{
		ers.push_back(x);
		ers.push_back(y);
		ers.push_back(z);
	};
	ERs ers;
	double slope;
	Rect box;
	string word;
};

struct Node
{
	Node(ER *v, const int i) : vertex(v), index(i){};
	ER* vertex;
	int index;
	vector<Node> adj_list;
	vector<double> edge_prob;
};

typedef vector<Node> Graph;

class ERFilter
{
public:
	ERFilter(int thresh_step = 2, int min_area = 100, int max_area = 100000, int stability_t = 2, double overlap_coef = 0.7);
	~ERFilter()	{}
	
	//! modules
	AdaBoost *adb1;
	AdaBoost *adb2;
	OCR *ocr;
	
	//! functions
	void text_detect(Mat src, ERs &root, vector<ERs> &pool, vector<ERs> &strong, vector<ERs> &weak, ERs &tracked, vector<Text> &text);
	void compute_channels(Mat &src, Mat &YCrcb, vector<Mat> &channels);
	ER* er_tree_extract(Mat input);
	void non_maximum_supression(ER *er, ERs &pool, Mat input);
	void classify(ERs &pool, ERs &strong, ERs &weak, Mat input);
	void er_delete(ER *er);
	void er_track(vector<ERs> &strong, vector<ERs> &weak, ERs &all_er, vector<Mat> &channel, Mat Ycrcb);
	void er_grouping(ERs &all_er, vector<Text> &text);
	void er_grouping_ocr(ERs &all_er, vector<Mat> &channel, vector<Text> &text, const double min_ocr_prob);
	vector<double> make_LBP_hist(Mat input, const int N = 2, const int normalize_size = 24);
	bool load_tp_table(const char* filename);
	Mat calc_LBP(Mat input, const int size = 24);

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
	void process_stack(const int new_pixel_grey_level, ERs &er_stack);

	// Gouping operation functions
	inline bool is_neighboring(ER *a, ER *b);
	inline bool is_overlapping(ER *a, ER *b);
	void build_graph(Text &text, Graph &graph);
	void solve_graph(Text &text, Graph &graph);
	void inner_suppression(ERs &pool);
	void overlap_suppression(ERs &pool);

	// feature extract
	Vec3d color_hist(Mat input);

	double tp[65][65];
};


class StrokeWidth
{
public:
	double SWT(Mat input);

private:
	struct SWTPoint2d {
		SWTPoint2d(int _x, int _y) : x(_x), y(_y) {};
		int x;
		int y;
		float SWT;
	};
	struct Ray {
		Ray(SWTPoint2d _p, SWTPoint2d _q, vector<SWTPoint2d> _points) : p(_p), q(_q), points(_points){};
		SWTPoint2d p;
		SWTPoint2d q;
		vector<SWTPoint2d> points;
	};
};


class ColorHist
{
public:
	ColorHist() { }
	inline void calc_hist(Mat img);
	inline double compare_hist(ColorHist ch);

	double c1[256];
	double c2[256];
	double c3[256];
};


double fitline_LSE(const vector<Point> &p);
double fitline_LMS(const vector<Point> &p);
double fitline_avgslope(const vector<Point> &p);
void calc_color(ER* er, Mat mask_channel, Mat color);
vector<vector<int> > comb(int N, int K);
double standard_dev(vector<double> arr, bool normalize);


#endif