#ifndef MULTITRAIN
#define MULTITRAIN

#include "util.h"

class Problem{
	public:
	map<string,int> label_index_map;
	vector<string> label_name_list;
	vector<SparseVec*> data;
	vector<Labels> labels;
	int N;//number of samples
	int D;//dimension
	int K;
};


class Param{
	public:
	char* trainFname;
	char* modelFname;
	double lambda; //for L1-norm (default 1/N)
	double C; //weight of loss
	int speed_up_rate; // speed up rate for sampling
	int split_up_rate; // split up [K] into a number of subsets	
	Problem* prob;
	
	//solver-specific param
	int solver;
	int max_iter;
	int max_select;
	bool using_importance_sampling;
	
	Param(){
		solver = 0;
		lambda = 1.0;
		C = 10.0;
		max_iter = 20;
		max_select = 1;
		speed_up_rate = 1;
		split_up_rate = 1;
		using_importance_sampling = false;
	}
};

class Model{
	public:
	Model(){
		label_name_list = new vector<string>();
		label_index_map = new map<string,int>();
	}
	Model(Problem* prob, HashVec** _w){
		D = prob->D;
		K = prob->K;
		label_name_list = &(prob->label_name_list);
		label_index_map = &(prob->label_index_map);
		w = _w;
	}
		
	int D;
	int K;
	HashVec** w;
	vector<string>* label_name_list;
	map<string,int>* label_index_map;
};

void readData(char* fname, Problem* prob)
{
	map<string,int>* label_index_map = &(prob->label_index_map);
	vector<string>* label_name_list = &(prob->label_name_list);
	
	ifstream fin(fname);
	char* line = new char[LINE_LEN];
	
	int d = -1;
	while( !fin.eof() ){
		
		fin.getline(line, LINE_LEN);
		string line_str(line);
		
		if( line_str.length() < 2 && fin.eof() )
			break;

		vector<string> tokens = split(line_str, " ");
		//get label index
		Labels lab_indices;
		lab_indices.clear();
		map<string,int>::iterator it;
		int st = 0;
		while (st < tokens.size() && tokens[st].find(":") == string::npos){
			// truncate , out
			while (*(tokens[st].end()-1) == ','){
				tokens[st].erase(tokens[st].end()-1);
			}
			if( (it=label_index_map->find(tokens[st])) == label_index_map->end() ){
				lab_indices.push_back(label_index_map->size());
				label_index_map->insert(make_pair(tokens[st],lab_indices.at(st)));
			}else{
				lab_indices.push_back(it->second);
			}
			st++;
		}
		
		SparseVec* ins = new SparseVec();
		for(int i=st;i<tokens.size();i++){
			vector<string> kv = split(tokens[i],":");
			int ind = atoi(kv[0].c_str());
			double val = atof(kv[1].c_str());
			ins->push_back(make_pair(ind,val));
			if( ind > d )
				d = ind;
		}
		
		prob->data.push_back(ins);
		prob->labels.push_back(lab_indices);
	}
	fin.close();

	prob->D = d+1; //adding bias
	prob->N = prob->data.size();
	prob->K = label_index_map->size();
	label_name_list->resize(prob->K);
	for(map<string,int>::iterator it=label_index_map->begin();
			it!=label_index_map->end();
			it++)
		(*label_name_list)[it->second] = it->first;
	
	delete[] line;
}

#endif
