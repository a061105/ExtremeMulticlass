#ifndef MULTITRAIN
#define MULTITRAIN

#include "util.h"

class Param{
	public:
	char* trainFname;
	char* modelFname;
	double lambda; //for L1-norm (default 1/N)
	double theta; //for L2-norm (default 1/N)
};

class Problem{
	public:
	map<string,int> label_index_map;
	vector<string> label_name_list;
	vector<SparseVec*> data;
	vector<int> labels;
	int N;//number of samples
	int D;//dimension
	int K;
};


#endif
