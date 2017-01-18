#include "adaboost.h"


//==================================================
//================= TrainingData ===================
//==================================================
const int TrainingData::get_num() { return nums; }
const int TrainingData::get_dim() { return dims; }

bool TrainingData::read_data(string filename)
{
	fstream fin;
	fin.open(filename, fstream::in);
	if (!fin.is_open())
	{
		cout << "Error: the input file is not opened!!" << endl;
		return false;
	}

	int i = 0;
	string buffer;
	while (getline(fin, buffer))			// fetch a row from the file
	{
		stringstream row(buffer);
		string item;

		data.push_back(FeatureVector());
		getline(row, item, ' ');
		data[i].label = stod(item);

		while (getline(row, item, ' '))
		{
			data[i].fv.push_back(stod(item));
		}
		i++;
	}

	nums = data.size();
	dims = data.front().fv.size();
	return true;
}


bool TrainingData::write_data(string filename)
{
	fstream fout;
	fout.open(filename, fstream::out);
	if (!fout.is_open())
	{
		cout << "Error: the output file is not opened!!" << endl;
		return false;
	}

	if (data.empty())
	{
		cout << "No data.";
		return false;
	}
		

	nums = data.size();
	dims = data.front().fv.size();

	for (int i = 0; i < data.size(); i++)
	{
		fout << data[i].label;

		for (int j = 0; j < data[i].fv.size(); j++)
		{
			fout << " " << data[i].fv[j];
		}

		fout << endl;
	}
}
//==================================================
//================= DecisionStump ==================
//==================================================
DecisionStump::DecisionStump(int _dim, int _dir, double t) : dim(_dim), dir(_dir), thresh(t) {}

double DecisionStump::predict(const vector<double> fv)
{
	return (fv[dim]*dir < thresh*dir) ? POS : NEG;
}

void DecisionStump::print_classifier()
{
	cout << "dim:" << dim << " ";
	cout << "dir:" << dir << " ";
	cout << "thresh:" << thresh << " ";
}

vector<double> DecisionStump::get_para()
{
	vector<double> para;
	para.push_back(dim);
	para.push_back(dir);
	para.push_back(thresh);
	return para;
}

void DecisionStump::set_para(const vector<double> para)
{
	if (para.size() != 3)
	{
		cout << "wrong dims";
		return;
	}

	dim = para[0];
	dir = para[1];
	thresh = para[2];
}


//==================================================
//================= DecisionStump ==================
//==================================================
RealDecisionStump::RealDecisionStump(int d, double t, double _cp, double _cn) : dim(d), thresh(t), cp(_cp), cn(_cn) {}

double RealDecisionStump::predict(const vector<double> fv)
{
	return (fv[dim] < thresh) ? cp : cn;
}

void RealDecisionStump::print_classifier()
{
	cout << "dim:" << dim << " ";
	cout << "thresh:" << thresh << " ";
	cout << "cp:" << cp << " ";
	cout << "cn:" << cn << endl;
}

vector<double> RealDecisionStump::get_para()
{
	vector<double> para;
	para.push_back(dim);
	para.push_back(thresh);
	para.push_back(cp);
	para.push_back(cn);
	return para;
}

void RealDecisionStump::set_para(const vector<double> para)
{
	if (para.size() != 4)
	{
		cout << "wrong dims";
		return;
	}

	dim = para[0];
	thresh = para[1];
	cp = para[2];
	cn = para[3];
}


//==================================================
//=================== Adaboost =====================
//==================================================
AdaBoost::AdaBoost(int boost, int base, int iter) : boost_type(boost), base_type(base), num_of_iter(iter) {}
AdaBoost::AdaBoost(string filename)
{
	load_classifier(filename);
}
AdaBoost::~AdaBoost()
{
	for (auto it : strong_classifier)
		delete it;
}
bool AdaBoost::set_boost_type(int _type)
{
	if (_type == DISCRETE || _type == REAL)
	{
		boost_type = _type;
		return true;
	}
		
	else
	{
		cout << "Invalid type" << endl;
		return false;
	}
}

bool AdaBoost::set_base_type(int _type)
{
	if (_type == DECISION_STUMP)
	{
		base_type = _type;
		return true;
	}
	else
	{
		cout << "Invalid type" << endl;
		return false;
	}
}

void AdaBoost::set_num_iter(int _type){ num_of_iter = _type; }
int AdaBoost::get_boost_type() { return boost_type; }
int AdaBoost::get_base_type() { return base_type; }
int AdaBoost::get_num_iter() { return num_of_iter; }

double AdaBoost::predict(vector<double> fv)
{
	double score = 0;
	for (int i = 0; i < strong_classifier.size(); i++)
	{
		score += strong_classifier[i]->predict(fv) * classifier_weight[i];
	}
	
	return score;
}

void AdaBoost::train_classifier(TrainingData &td, string outfile)
{
	strong_classifier = vector<BaseClassifier*>();

	if (boost_type == REAL)
		real_training(td);

	write_classifier(outfile);
}

void AdaBoost::real_training(TrainingData &td)
{
	const int dims = td.get_dim();
	const int nums = td.get_num();

	double Pr_p = .0;
	double Pr_n = .0;
	double Pw_p = .0;
	double Pw_n = .0;
	double current_Z = .0;
	double min_Z = DBL_MAX;
	double c_p;
	double c_n;
	double epsilon = 1.0 / (4.0 * nums);
	double thresh;
	int dim;
	int dir = 1;


	// make threshold set (element of set is unique)
	vector<set<double> > thresh_set(dims);
	for (int i = 0; i < dims; i++)
	{
		vector<double> column_data(nums);
		for (int j = 0; j < nums; j++)
		{
			column_data[j] = td.data[j].fv[i];
		}
		thresh_set[i] = set<double>(column_data.begin(), column_data.end());
	}
	



	// initialize feature weight
	vector<double>weight(nums, 1.0 / nums);
	int pos = 0;
	int neg = 0;
	for (auto it : td.data)
	{
		if (it.label == POS)
			pos++;
		else
			neg++;
	}
	for (int j = 0; j < nums; j++)
	{
		if (td.data[j].label == POS)
			weight[j] = 0.5 / pos;
		else
			weight[j] = 0.5 / neg;
	}



	for (int t = 0; t < num_of_iter; t++)
	{
		min_Z = DBL_MAX;

		// search all the weak classifiers
		for (int i = 0; i < dims; i++)
		{
			for (set<double>::iterator it = thresh_set[i].begin(); it != thresh_set[i].end(); it++)
			{
				Pr_p = .0;
				Pr_n = .0;
				Pw_p = .0;
				Pw_n = .0;

				for (int j = 0; j < nums; j++)
				{
					if (td.data[j].fv[i] < *it && td.data[j].label == POS)
						Pr_p += weight[j];
					else if (td.data[j].fv[i] < *it && td.data[j].label == NEG)
						Pw_n += weight[j];
					else if (td.data[j].fv[i] >= *it && td.data[j].label == POS)
						Pw_p += weight[j];
					else if (td.data[j].fv[i] >= *it && td.data[j].label == NEG)
						Pr_n += weight[j];
				}

				current_Z = 2 * (sqrt(Pr_p*Pw_n) + sqrt(Pr_n*Pw_p));
				if (current_Z < min_Z)
				{
					min_Z = current_Z;
					dim = i;
					thresh = *it;

					c_p = 0.5 * log((Pr_p + epsilon) / (Pw_n + epsilon));
					c_n = 0.5 * log((Pw_p + epsilon) / (Pr_n + epsilon));
				}
			}
		}

		strong_classifier.push_back(new RealDecisionStump(dim, thresh, c_p, c_n));
		classifier_weight.push_back(1.0);

		// update weight of exmaples, Z is the normalize factor
		for (int j = 0; j < nums; j++)
		{
			double new_probability;

			if (td.data[j].fv[dim] < thresh)
				new_probability = weight[j] * exp(-1 * td.data[j].label*c_p) / min_Z;
			else
				new_probability = weight[j] * exp(-1 * td.data[j].label*c_n) / min_Z;

			weight[j] = new_probability;
		}

		
		double result_threshold = 2.0;
		double FP = 0;
		double FN = 0;
		double TP = 0;
		double TN = 0;
		for (int j = 0; j < nums; j++)
		{
			double result = predict(td.data[j].fv);
			if (result >= result_threshold)
			{
				if (td.data[j].label == POS)
					TP++;
				else
					FP++;
			}
			else
			{
				if (td.data[j].label == POS)
					FN++;
				else
					TN++;
			}
		}
		double training_error = (FP+FN) / nums;
		double precision = TP / (TP + FP);
		double recall = TP / (TP + FN);
		
		cout << setw(1);
		cout << t << "  Z= " << min_Z << "  Dim: " << dim << "  Thresh: " << thresh << "  Cp: " << c_p << "  Cn: " << c_n;
		cout << "  Training error: " << training_error << "  Precision: " << precision << "  Recall: " << recall << endl;
	}
}

bool AdaBoost::load_classifier(string filename)
{
	fstream fin;
	fin.open(filename, fstream::in);
	if (!fin.is_open())
	{
		cout << "Error: the input file is not opened!!" << endl;
		return false;
	}

	strong_classifier.clear();

	string buffer;
	fin >> buffer;
	if (buffer == "boost_type")
	{
		fin >> buffer;
		boost_type = (buffer == "DISCRETE") ? DISCRETE : REAL;
	}
	fin >> buffer;
	if (buffer == "base_type")
	{
		fin >> buffer;
		base_type = (buffer == "DECISION_STUMP") ? DECISION_STUMP : -1;
	}
	fin >> buffer;
	if (buffer == "num_of_iter")
	{
		fin >> buffer;
		num_of_iter = stoi(buffer);
	}

	getline(fin, buffer);		// avoid bug

	while (getline(fin, buffer))
	{
		stringstream row(buffer);
		string item;
		vector<double> para;

		getline(row, item, ' ');
		classifier_weight.push_back(stod(item));

		while (getline(row, item, ' '))
		{
			para.push_back(stod(item));
		}
		
		BaseClassifier *bc;
		if (boost_type == DISCRETE)
			bc = new DecisionStump();
		else
			bc = new RealDecisionStump();

		bc->set_para(para);
		strong_classifier.push_back(bc);
	}
}

bool AdaBoost::write_classifier(string filename)
{
	fstream fout;
	fout.open(filename, fstream::out);
	if (!fout.is_open())
	{
		cout << "Error: the output file is not opened!!" << endl;
		return false;
	}

	string s_boost_type = (boost_type == DISCRETE) ? "DISCRETE" : "REAL";
	string s_base_type = (base_type == DECISION_STUMP) ? "DECISION_STUMP" : "";

	fout << "boost_type " << s_boost_type << endl;
	fout << "base_type " << s_base_type << endl;
	fout << "num_of_iter " << num_of_iter << endl;
	
	for (int i = 0; i < strong_classifier.size(); i++)
	{
		fout << classifier_weight[i] << " ";

		vector<double> para = strong_classifier[i]->get_para();
		for (int j = 0; j < para.size(); j++)
		{
			fout << para[j] << " ";
		}
		fout << endl;
	}
}


void AdaBoost::print_classifier()
{
	for (int i = 0; i < strong_classifier.size(); i++)
	{
		cout << "Classifier " << i << "\t";
		strong_classifier[i]->print_classifier();
	}
}