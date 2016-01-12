#ifndef MULTITRAIN
#define MULTITRAIN

#include "util.h"
#include "newHash.h"

class Problem{
	public:
	static map<string,int> label_index_map;
	static vector<string> label_name_list;
	static int D;//dimension
	static int K;
	
	vector<SparseVec*> data;
	vector<Labels> labels;
	int N;//number of samples
};

map<string,int> Problem::label_index_map;
vector<string> Problem::label_name_list;
int Problem::D = -1;
int Problem::K = -1;

class HeldoutEval{
	public:
	HeldoutEval(Problem* _heldout){
		heldout = _heldout;
		N = heldout->data.size();
		D = heldout->D;
		K = heldout->K;
		prod = new float_type[K];
		max_indices = new int[K];
		inside = new bool[K];
		for (int k = 0; k < K; k++)
			inside[k] = false;

		T = 1;
	}
	
	#ifdef USING_HASHVEC
	double calcAcc(pair<int, float_type>** v, int* size_v, int* hashindices, float_type lambda){
		hit=0.0;
		margin_hit = 0.0;
		for(int i=0;i<heldout->N;i++){
			memset(prod, 0.0, sizeof(float_type)*K);

			SparseVec* xi = heldout->data.at(i);
			Labels* yi = &(heldout->labels.at(i));
			int top = T;
			if (top == -1)
				top = yi->size();
			for (int t = 0; t <= top; t++)
				max_indices[t] = -1;
			for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){

				int j= it->first;
				float_type xij = it->second;
				if( j >= D )
					continue;
				pair<int, float_type>* vj = v[j];
				int size_vj0 = size_v[j] - 1;
				for (int k = 0; k < K; k++){
					int index_v = 0;
					find_index(vj, index_v, k, size_vj0, hashindices);
					float_type wjk = prox_l1(vj[index_v].second, lambda);
					prod[k] += wjk * xij;
					update_max_indices(max_indices, prod, k, top); 
				}
			}
			if (max_indices[0] == -1 || prod[max_indices[0]] < 0.0){
				for (int t = 0; t < top; t++){
					for (int k = 0; k < K; k++){
						if (prod[k] == 0.0){
							if (update_max_indices(max_indices, prod, k, top))
								break;
						}
					}
				}
			}
			for(int k=0;k<top;k++){
				bool flag = false;
				for (int j = 0; j < yi->size(); j++){
					if (yi->at(j) == max_indices[k]){
						flag = true;
					}
				}
				if (flag)
					hit += 1.0/top;
			}
		}
		return hit/N;
	}
	#else
	double calcAcc(pair<float_type, float_type>** v){
		hit=0.0;
		margin_hit = 0.0;
		for(int i=0;i<N;i++){
			memset(prod, 0.0, sizeof(float_type)*K);
				
			SparseVec* xi = heldout->data.at(i);
			Labels* yi = &(heldout->labels.at(i));
			
			int top = T;
			if (top == -1)
				top = yi->size();
			for (int t = 0; t < top; t++)
				max_indices[t] = -1;
			for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){

				int j= it->first;
				float_type xij = it->second;
				if( j >= D )
					continue;
				pair<float_type, float_type>* vj = v[j];
				for (int k = 0; k < K; k++){	
					prod[k] += vj[k].second * xij;
					update_max_indices(max_indices, prod, k, top);
				}
			}
			if (max_indices[0] == -1 || prod[max_indices[0]] < 0.0){
				for (int t = 0; t < top; t++){
					for (int k = 0; k < K; k++){
						if (prod[k] == 0.0){
							if (update_max_indices(max_indices, prod, k, top))
								break;
						}
					}
				}
			}
			for(int k=0;k<top;k++){
				bool flag = false;
				for (int j = 0; j < yi->size(); j++){
					if (yi->at(j) == max_indices[k]){
						flag = true;
					}
				}
				if (flag)
					hit += 1.0/top;
			}
		}
		return hit/N;
	}
	#endif
	
	#ifdef USING_HASHVEC
	double calcAcc(pair<int, pair<float_type, float_type>>** v, int* size_v, vector<int>*** nnz_index, int*& hashindices, int split_up_rate){
	#else
	double calcAcc(pair<float_type, float_type>** v, vector<int>*** nnz_index, int split_up_rate){
	#endif
		vector<SparseVec*>* data = &(heldout->data);
		vector<Labels>* labels = &(heldout->labels);
		hit=0.0;
		margin_hit = 0.0;
		for(int i=0;i<heldout->N;i++){
			memset(prod, 0.0, sizeof(float_type)*K);
			
			SparseVec* xi = data->at(i);
			Labels* yi = &(labels->at(i));
			
			int top = T;
			if (top == -1)
				top = yi->size();
			for (int tt = 0; tt < top; tt++)
				max_indices[tt] = -1;
			for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){

				int j= it->first;
				float_type xij = it->second;
				if( j >= D )
					continue;
				#ifdef USING_HASHVEC
				pair<int, pair<float_type, float_type>>* vj = v[j];
				int size_vj0 = size_v[j] - 1;
				#else
				pair<float_type, float_type>* vj = v[j];
				#endif
				for (int S = 0; S < split_up_rate; S++){
					vector<int>* wjS = nnz_index[j][S];
					for (vector<int>::iterator it2 = wjS->begin(); it2 != wjS->end(); it2++){
						int k = *it2;
						#ifdef USING_HASHVEC
						int index_v = 0;
						find_index(vj, index_v, k, size_vj0, hashindices);
						float_type wjk = vj[index_v].second.second;
						#else
						float_type wjk = vj[k].second;
						#endif
						if (wjk == 0.0 || inside[k]){
							*it2=*(wjS->end()-1);
							wjS->erase(wjS->end()-1);
							it2--;
							continue; 
						}
						inside[k] = true;
							
						prod[k] += wjk * xij;
						update_max_indices(max_indices, prod, k, top);
					}
					for (vector<int>::iterator it2 = wjS->begin(); it2 != wjS->end(); it2++){
						inside[*it2] = false;
					}
				}
			}
			if (max_indices[0] == -1 || prod[max_indices[0]] < 0.0){
				for (int t = 0; t < top; t++){
					for (int k = 0; k < K; k++){
						if (prod[k] == 0.0){
							if (update_max_indices(max_indices, prod, k, top))
								break;
						}
					}
				}
			}
			for(int k=0;k<top;k++){
				bool flag = false;
				for (int j = 0; j < yi->size(); j++){
					if (yi->at(j) == max_indices[k]){
						flag = true;
					}
				}
				if (flag)
					hit += 1.0/top;
			}
		}
		return hit/N;
	}
	private:
	int N,D,K;
	Problem* heldout;
	float_type* prod;
	int* max_indices;
	float_type hit, margin_hit;
	bool* inside;
	int T;
};
class Param{
	public:
	char* trainFname;
	char* modelFname;
	char* heldoutFname;
	float_type lambda; //for L1-norm (default 1/N)
	float_type C; //weight of loss
	int speed_up_rate; // speed up rate for sampling
	int split_up_rate; // split up [K] into a number of subsets	
	Problem* train;
	HeldoutEval* heldoutEval = NULL;
	//solver-specific param
	int solver;
	int max_iter;
	int max_select;
	bool using_importance_sampling;
	int post_solve_iter;

	Param(){
		solver = 0;
		lambda = 1.0;
		C = 1.0;
		max_iter = 20;
		max_select = 1;
		speed_up_rate = -1;
		split_up_rate = 1;
		using_importance_sampling = false;
		post_solve_iter = 0;

		heldoutFname == NULL;
		train = NULL;
	}
};

class Model{
	public:
	Model(){
		label_name_list = new vector<string>();
		label_index_map = new map<string,int>();
	}

	#ifdef USING_HASHVEC
	Model(Problem* prob, vector<int>** _w_hash_nnz_index, pair<int, float_type>** _w, int* _size_w, int* _hashindices){
		D = prob->D;
		K = prob->K;
		label_name_list = &(prob->label_name_list);
		label_index_map = &(prob->label_index_map);
		w_hash_nnz_index = _w_hash_nnz_index;
		w = _w;
		size_w = _size_w;
                hashindices = _hashindices;
	}
	pair<int, float_type>** w;
	int* size_w;
	int* hashindices;
	#else
	Model(Problem* prob, vector<int>** _w_hash_nnz_index, float_type** _w){
		D = prob->D;
		K = prob->K;
		label_name_list = &(prob->label_name_list);
		label_index_map = &(prob->label_index_map);
		w_hash_nnz_index = _w_hash_nnz_index;
		w = _w;
	}
	float_type** w;
	#endif

	HashVec** Hw;
	vector<int>** w_hash_nnz_index;	
	int D;
	int K;
	vector<string>* label_name_list;
	map<string,int>* label_index_map;
};

class StaticModel{

	public:
	StaticModel(){
		label_name_list = new vector<string>();
		label_index_map = new map<string,int>();
	}
	SparseVec* w;
	int D;
	int K;
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
			if (tokens[st].size() == 0){
				st++;
				continue;
			}
			if (*(tokens[st].end()-1) == ','){
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
			float_type val = atof(kv[1].c_str());
			ins->push_back(make_pair(ind,val));
			if( ind > d )
				d = ind;
		}
		
		prob->data.push_back(ins);
		prob->labels.push_back(lab_indices);
	}
	fin.close();
	
	if (prob->D < d+1){
		prob->D = d+1; //adding bias
	}
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
