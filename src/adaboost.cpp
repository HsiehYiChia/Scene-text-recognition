#include "../inc/adaboost.h"


//==================================================
//================= TrainingData ===================
//==================================================
const int TrainingData::get_num() { return nums; }
const int TrainingData::get_dim() { return dims; }
void TrainingData::set_num(int n) { nums = n; }
void TrainingData::set_dim(int d) { dims = d; }

bool TrainingData::read_data(string filename)
{
	fstream fin;
	fin.open(filename, fstream::in);
	if (!fin.is_open())
	{
		std::cout << "Error: " << filename << " is not opened!!" << endl;
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
		std::cout << "Error: the output file is not opened!!" << endl;
		return false;
	}

	if (data.empty())
	{
		std::cout << "No data.";
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

inline double DecisionStump::predict(const vector<double> fv)
{
	return (fv[dim]*dir < thresh*dir) ? POS : NEG;
}

void DecisionStump::print_classifier()
{
	std::cout << "dim:" << dim << " ";
	std::cout << "dir:" << dir << " ";
	std::cout << "thresh:" << thresh << " ";
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
		std::cout << "wrong dims";
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

inline double RealDecisionStump::predict(const vector<double> fv)
{
	return (fv[dim] < thresh) ? cp : cn;
}

void RealDecisionStump::print_classifier()
{
	std::cout << "dim:" << dim << " ";
	std::cout << "thresh:" << thresh << " ";
	std::cout << "cp:" << cp << " ";
	std::cout << "cn:" << cn << endl;
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
		std::cout << "wrong dims";
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
		std::cout << "Invalid type" << endl;
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
		std::cout << "Invalid type" << endl;
		return false;
	}
}

void AdaBoost::set_num_iter(int _iter){ num_of_iter = _iter; }
int AdaBoost::get_boost_type() { return boost_type; }
int AdaBoost::get_base_type() { return base_type; }
int AdaBoost::get_num_iter() { return num_of_iter; }

double AdaBoost::predict(vector<double> fv)
{
	double score = 0;
	if (boost_type == DISCRETE)
	{
		for (int i = 0; i < strong_classifier.size(); i++)
			score += strong_classifier[i]->predict(fv) * classifier_weight[i];
	}
	else
	{
		for (int i = 0; i < strong_classifier.size(); i++)
			score += strong_classifier[i]->predict(fv);
	}
	
	
	return score;
}

void AdaBoost::train_classifier(TrainingData &td, string outfile)
{
	strong_classifier = vector<BaseClassifier*>();

	if (boost_type == DISCRETE)
		discrete_training(td);
	else if (boost_type == REAL)
		real_training(td);

	write_classifier(outfile);
}


void AdaBoost::discrete_training(TrainingData &td)
{
	const int dims = td.get_dim();
	const int nums = td.get_num();

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
#pragma omp parallel for
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
			if (td.data[j].fv[dim] < thresh)
				weight[j] = weight[j] * exp(-1 * td.data[j].label*c_p) / min_Z;
			else
				weight[j] = weight[j] * exp(-1 * td.data[j].label*c_n) / min_Z;
		}

		
		double result_threshold = 0.0;
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
		
		std::cout << setw(1);
		std::cout << t << "  Z= " << min_Z << "  Dim: " << dim << "  Thresh: " << thresh << "  Cp: " << c_p << "  Cn: " << c_n;
		std::cout << "  Training error: " << training_error << "  Precision: " << precision << "  Recall: " << recall << endl;
	}
}

bool AdaBoost::load_classifier(string filename)
{
	fstream fin;
	fin.open(filename, fstream::in);
	if (!fin.is_open())
	{
		std::cout << "Error: " << filename << " is not opened!!" << endl;
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
		std::cout << "Error: the output file is not opened!!" << endl;
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
		std::cout << "Classifier " << i << "\t";
		strong_classifier[i]->print_classifier();
	}
}


//==================================================
//=============== Cascade Adaboost =================
//==================================================
CascadeBoost::CascadeBoost() { }
CascadeBoost::CascadeBoost(string filename)
{
	load_classifier(filename);
}
CascadeBoost::CascadeBoost(int boost, int base, double _Ftarget, double _f, double _d) : boost_type(boost), base_type(base), Ftarget(_Ftarget), f(_f), d(_d)  {}

void CascadeBoost::set_num_iter(int _iter) { ; }
int CascadeBoost::get_num_iter() { return classifier.size(); }

double CascadeBoost::predict(vector<double> fv)
{
	double score_stage = 0;
	int offset = 0;
	if (boost_type == DISCRETE)
	{
		for (int i = 0; i < num_of_iter.size(); i++)
		{
			score_stage = 0;
			for (int j = offset; j < offset + num_of_iter[i]; j++)
				score_stage += classifier[j]->predict(fv) * classifier_weight[j];

			if (score_stage < thresh[i])
				return -DBL_MAX;
			else
				offset += num_of_iter[i];
		}
	}

	else if (boost_type == REAL)
	{
		for (int i = 0; i < num_of_iter.size(); i++)
		{
			score_stage = 0;
			for (int j = offset; j < offset + num_of_iter[i]; j++)
				score_stage += classifier[j]->predict(fv);

			if (score_stage < thresh[i])
				return -DBL_MAX;
			else
				offset += num_of_iter[i];
		}
	}
	
	return score_stage;
}


void CascadeBoost::train_classifier(TrainingData &td, string outfile)
{
	classifier = vector<BaseClassifier*>();

	// initialize feature weight
	double Fi = 1.0;
	double Di = 1.0;
	int i = 0;

	int p_num = 0;
	int n_num = 0;
	for (auto it : td.data)
	{
		(it.label == POS) ? p_num++ : n_num++;
	}

	int offset = 0;
	while (Fi > Ftarget)
	{
		const int nums = td.get_num();
		const int dims = td.get_dim();

		// make threshold set (element of set is unique) and sorted data
		vector<set<double>> thresh_set(dims);
		vector<vector<ColFea>> sorted_data(dims, vector<ColFea>(nums));
#pragma omp parallel for
		for (int m = 0; m < dims; m++)
		{
			vector<double> column_data(nums);
			for (int n = 0; n < nums; n++)
			{
				column_data[n] = td.data[n].fv[m];
				sorted_data[m][n].f = td.data[n].fv[m];
				sorted_data[m][n].idx = n;
				sorted_data[m][n].label = td.data[n].label;
			}
			thresh_set[m] = set<double>(column_data.begin(), column_data.end());
			sort(sorted_data[m].begin(), sorted_data[m].end(), [](ColFea a, ColFea b) { return a.f < b.f; });
		}



		vector<double>weight(nums, 1.0 / nums);
		int pos = 0;
		int neg = 0;
		for (auto it : td.data)
		{
			(it.label == POS) ? pos++ : neg++;
		}
		for (int j = 0; j < nums; j++)
		{
			(td.data[j].label == POS) ? weight[j] = 0.5 / pos :	weight[j] = 0.5 / neg;
		}
		

		i++;
		int ni = 0;
		double Ti = .0;
		double F_prev = Fi;
		double D_prev = Di;
		num_of_iter.push_back(ni);
		thresh.push_back(Ti);

		std::cout << "Layer " << i << "    pos=" << pos << " neg=" << neg << endl;
		while (Fi > f * F_prev && Fi > Ftarget)
		{
			// Use P and N to train a classifier with ni features using AdaBoost
			ni++;
			if (boost_type == DISCRETE)
				discrete_training(td, thresh_set, sorted_data, weight);
			else if (boost_type == REAL)
				real_training(td, thresh_set, sorted_data, weight);
			else if (boost_type == GENTLE)
				gentle_training(td, thresh_set, sorted_data, weight);

			// Evaluate current cascaded classifier on validation set to 
			// determine Fi and Di
			// Decrease threshold for the ith classifier until the current
			// cascaded classifier has a detection rate of at least
			Ti = 0.5;
			num_of_iter.back() = ni;
			double Pi;
			do
			{
				Ti -= 0.5;
				thresh.back() = Ti;
				int tp = 0;
				int fp = 0;

				#pragma omp parallel for
				for (int i = 0; i < td.data.size(); i++)
				{
					if (predict(td.data[i].fv) > Ti)
					{
						(td.data[i].label == POS) ? tp++ : fp++;
					}
				}
				Di = (double)tp / p_num;
				Fi = (double)fp / n_num;
				Pi = (double)tp / (tp + fp);
			} while (Di < d*D_prev);

			std::cout << offset + ni << " Fi=" << Fi << " Di=" << Di << " Ti=" << Ti << " Pi=" << Pi << endl;
			if (offset + ni > 5000)
			{
				goto end;
			}
		}
		offset += ni;
		


		// if Fi > Ftarget then evaluate the current cascaded detector on
		// the set of non - face images and put any false detections into
		// the set N
		if (Fi > Ftarget)
		{
			TrainingData td_tmp = TrainingData();
			for (auto it : td.data)
			{
				if (it.label == POS)
					td_tmp.data.push_back(it);
				else if (it.label == NEG)
				{
					if (predict(it.fv) > thresh.back())
						td_tmp.data.push_back(it);
				}
			}
			
			td = td_tmp;
			td.set_num(td.data.size());
			td.set_dim(td.data[0].fv.size());
		}
	}
end:
	write_classifier(outfile);
}



void CascadeBoost::discrete_training(TrainingData &td, vector<set<double>> &thresh_set, vector<vector<ColFea>> &sorted_data, vector<double> &weight)
{
	const int dims = td.get_dim();
	const int nums = td.get_num();

	double total_pos_weight = .0;
	double total_neg_weight = .0;
	double current_pos_weight = .0;
	double current_neg_weight = .0;
	double lowest_err = 1;
	double current_err;
	double thresh;
	unsigned int dim;
	int dir;


	// get total weight for both pos and neg
	for (int j = 0; j < nums; j++)
	{
		if (td.data[j].label == POS)
			total_pos_weight += weight[j];
		else
			total_neg_weight += weight[j];
	}

	// traverse the feature and find out which feature and threshold is the best
	for (int i = 0; i < dims; i++)
	{
		current_pos_weight = .0;
		current_neg_weight = .0;

		int j = 0;
		for (auto it = thresh_set[i].begin(); it != thresh_set[i].end(); it++)
		{
			while (j < nums && sorted_data[i][j].f < *it)
			{
				int idx = sorted_data[i][j].idx;
				if (sorted_data[i][j].label == POS)
					current_pos_weight += weight[idx];
				else
					current_neg_weight += weight[idx];

				j++;
			}

			current_err = min(current_pos_weight + total_neg_weight - current_neg_weight,
								current_neg_weight + total_pos_weight - current_pos_weight);

			if (current_err < lowest_err)
			{
				// dir = 1,  for err = S- + (T+ - S+)
				// dir = -1, for err = S+ + (T- - S-)
				lowest_err = current_err;
				thresh = sorted_data[i][j].f;
				dim = i;
				dir = (current_pos_weight + total_neg_weight - current_neg_weight <	current_neg_weight + total_pos_weight - current_pos_weight) ? -1 : 1;
			}
		}
	}


	// get the best weak classifier at this iteration
	classifier_weight.push_back( log((1 - lowest_err) / lowest_err) );
	classifier.push_back(new DecisionStump(dim, dir, thresh));


	// update weight of exmaples
	for (int j = 0; j < nums; j++)
	{
		if (td.data[j].fv[dim]*dir < thresh*dir)
		{
			int idx = sorted_data[dim][j].idx;
			weight[idx] = weight[idx] * lowest_err / (1 - lowest_err);
		}
	}

	// once we update the weight, we need to normalize the weight
	double normalize_factor = 0;
	for (int j = 0; j < nums; j++)
		normalize_factor += weight[j];

	for (int j = 0; j < nums; j++)
		weight[j] /= normalize_factor;
}


void CascadeBoost::real_training(TrainingData &td, vector<set<double>> &thresh_set, vector<vector<ColFea>> &sorted_data, vector<double> &weight)
{
	const int dims = td.get_dim();
	const int nums = td.get_num();
	
	omp_lock_t update_lock;
	omp_init_lock(&update_lock);


	double total_pos_weight = .0;
	double total_neg_weight = .0;
	double min_Z = DBL_MAX;
	double c_p;
	double c_n;
	double epsilon = 1.0 / (4.0 * nums);
	double thresh;
	int dim;

	min_Z = DBL_MAX;

	// get total weight for both pos and neg
	for (int j = 0; j < nums; j++)
	{
		if (td.data[j].label == POS)
			total_pos_weight += weight[j];
		else
			total_neg_weight += weight[j];
	}

	// search all the weak classifiers
#pragma omp parallel for
	for (int i = 0; i < dims; i++)
	{
		double Pr_p = .0;
		double Pr_n = .0;
		double Pw_p = .0;
		double Pw_n = .0;
		double current_Z = .0;
		for (set<double>::iterator it = thresh_set[i].begin(); it != thresh_set[i].end(); it++)
		{
			Pr_p = .0;
			Pr_n = total_neg_weight;
			Pw_p = total_pos_weight;
			Pw_n = .0;

			int j = 0;
			while (j < nums && sorted_data[i][j].f < *it)
			{
				int idx = sorted_data[i][j].idx;
				if (sorted_data[i][j].label == POS)
				{
					Pr_p += weight[idx];
					Pw_p -= weight[idx];
				}
				else
				{
					Pw_n += weight[idx];
					Pr_n -= weight[idx];
				}

				j++;
			}

			current_Z = 2 * (sqrt(Pr_p*Pw_n) + sqrt(Pr_n*Pw_p));
			if (current_Z < min_Z)
			{
				omp_set_lock(&update_lock);
				if (current_Z < min_Z)
				{
					min_Z = current_Z;
					dim = i;
					thresh = *it;

					c_p = 0.5 * log((Pr_p + epsilon) / (Pw_n + epsilon));
					c_n = 0.5 * log((Pw_p + epsilon) / (Pr_n + epsilon));
				}
				omp_unset_lock(&update_lock);
			}
			
		}
	}

	classifier.push_back(new RealDecisionStump(dim, thresh, c_p, c_n));
	classifier_weight.push_back(1.0);

	// update weight of exmaples, Z is the normalize factor
	for (int j = 0; j < nums; j++)
	{
		if (td.data[j].fv[dim] < thresh)
			weight[j] = weight[j] * exp(-1 * td.data[j].label*c_p) / min_Z;
		else
			weight[j] = weight[j] * exp(-1 * td.data[j].label*c_n) / min_Z;
	}
}


void CascadeBoost::gentle_training(TrainingData &td, vector<set<double>> &thresh_set, vector<vector<ColFea>> &sorted_data, vector<double> &weight)
{

}


bool CascadeBoost::load_classifier(string filename)
{
	fstream fin;
	fin.open(filename, fstream::in);
	if (!fin.is_open())
	{
		std::cout << "Error: " << filename << " is not opened!!" << endl;
		return false;
	}

	classifier.clear();

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
		while (1)
		{
			fin >> buffer;
			try {
				num_of_iter.push_back(stod(buffer));
			} 
			catch (std::invalid_argument){
				break;
			}
		}
	}

	if (buffer == "threshold")
	{
		for (int j = 0; j < num_of_iter.size(); j++)
		{
			fin >> buffer;
			thresh.push_back(stod(buffer));
		}
			
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
		classifier.push_back(bc);
	}
}


bool CascadeBoost::write_classifier(string filename)
{
	fstream fout;
	fout.open(filename, fstream::out);
	if (!fout.is_open())
	{
		std::cout << "Error: the output file is not opened!!" << endl;
		return false;
	}

	string s_boost_type = (boost_type == DISCRETE) ? "DISCRETE" : "REAL";
	string s_base_type = (base_type == DECISION_STUMP) ? "DECISION_STUMP" : "";

	fout << "boost_type " << s_boost_type << endl;
	fout << "base_type " << s_base_type << endl;
	fout << "num_of_iter";
	for (auto it : num_of_iter)
		fout << " " << it;
	fout << endl;

	fout << "threshold";
	for (auto it : thresh)
		fout << " " << it;
	fout << endl;


	for (int i = 0; i < classifier.size(); i++)
	{
		fout << classifier_weight[i] << " ";

		vector<double> para = classifier[i]->get_para();
		for (int j = 0; j < para.size(); j++)
		{
			fout << para[j] << " ";
		}
		fout << endl;
	}
}


void CascadeBoost::print_classifier()
{
	int offset = 0;
	for (int i = 0; i < num_of_iter.size(); i++)
	{
		std::cout << "Layer " << i << endl;

		for (int j = offset; j < offset + num_of_iter[i]; j++)
		{
			std::cout << "Classifier " << j << "\t";
			classifier[j]->print_classifier();
		}
	}
}