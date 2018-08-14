#ifndef __MY_ADABOOST__
#define __MY_ADABOOST__

#define POS 1
#define NEG -1

#include <iostream>
#include <iomanip>      // std::setw
#include <fstream>	// open file
#include <istream>	// getline
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <math.h>
#include <cfloat>

#include <thread>
#include <omp.h>


using namespace std;

struct ColFea
{
	int label;
	int idx;
	double f;
};

struct FeatureVector
{
	FeatureVector() {}
	FeatureVector(int dims) { fv.resize(dims); }
	vector<double> fv;
	int label;			
};

class TrainingData
{
public:
	TrainingData() {};
	~TrainingData() {};

	// getters
	const int get_num();
	const int get_dim();

	// setters
	void set_num(int n);
	void set_dim(int d);

	// functions
	bool read_data(string filename);
	bool write_data(string filename);

	// feature vector
	vector<FeatureVector> data;

private:
	int nums;		// number of data
	int dims;		// dimention of feature vector, equals to the size of FeatureVector
};


class BaseClassifier
{
public:
	BaseClassifier() {};
	~BaseClassifier() {};
	virtual double predict(const vector<double> fv) = 0;
	virtual void print_classifier() = 0;
	virtual vector<double> get_para() = 0;
	virtual void set_para(const vector<double> para) = 0;
};

class DecisionStump : public BaseClassifier
{
public:
	DecisionStump() {};
	DecisionStump(int _dim, int _dir, double t);
	~DecisionStump() {};

	double predict(const vector<double> fv);
	void print_classifier();
	vector<double> get_para();
	void set_para(const vector<double> para);

private:
	int dim;
	int dir;
	double thresh;
};

class RealDecisionStump : public BaseClassifier
{
public:
	RealDecisionStump() {};
	RealDecisionStump(int d, double t, double _cp, double _cn);
	~RealDecisionStump() {};

	inline double predict(const vector<double> fv);
	void print_classifier();
	vector<double> get_para();
	void set_para(const vector<double> para);
	
private:
	int dim;
	double thresh;
	double cp;			// for real adaboost, Ct+
	double cn;			// for real adaboost, Ct-
};



class AdaBoost	
{
public:
	AdaBoost() {};
	AdaBoost(string filename);
	AdaBoost(int boost, int base, int iter);
	~AdaBoost();

	bool set_boost_type(int _type);
	bool set_base_type(int _type);
	virtual void set_num_iter(int _iter);
	int get_boost_type();
	int get_base_type();
	virtual int get_num_iter();
	virtual double predict(vector<double> fv);
	virtual void train_classifier(TrainingData &training_data, string outfile);
	virtual bool load_classifier(string filename);
	virtual bool write_classifier(string filename);
	virtual void print_classifier();
	enum {
		DISCRETE,
		REAL,
		GENTLE,
		DECISION_STUMP
	};

private:
	int boost_type;
	int base_type;
	int num_of_iter;
	vector<BaseClassifier*> strong_classifier;
	vector<double> classifier_weight;

	void discrete_training(TrainingData &td);
	void real_training(TrainingData &td);
};


class CascadeBoost : public AdaBoost
{
public:
	CascadeBoost();
	CascadeBoost(string filename);
	CascadeBoost(int boost, int base, double _Ftarget, double _f, double _d);
	void set_num_iter(int _iter);
	int get_num_iter();
	double predict(vector<double> fv);
	void train_classifier(TrainingData &td, string outfile);
	bool load_classifier(string filename);
	bool write_classifier(string filename);
	void print_classifier();


private:
	int boost_type;
	int base_type;
	int cascade_stages;
	double Ftarget;
	double f;
	double d;
	vector<BaseClassifier*> classifier;
	vector<int> num_of_iter;
	vector<int> thresh;
	vector<double> classifier_weight;
	
	void discrete_training(TrainingData &td, vector<set<double>> &thresh_set, vector<vector<ColFea>> &sorted_data, vector<double> &weight);
	void real_training(TrainingData &td, vector<set<double>> &thresh_set, vector<vector<ColFea>> &sorted_data, vector<double> &weight);
	void gentle_training(TrainingData &td, vector<set<double>> &thresh_set, vector<vector<ColFea>> &sorted_data, vector<double> &weight);
};


#endif