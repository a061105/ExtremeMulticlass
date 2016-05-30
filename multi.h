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
		prod = new Float[K];
		max_indices = new int[K];
		for (int k = 0; k < K; k++)
			max_indices[k] = k;
		inside = new bool[K];
		for (int k = 0; k < K; k++)
			inside[k] = false;

		T = 1;
	}

	~HeldoutEval(){
		delete[] max_indices;
		delete[] inside;
		delete[] prod;
	}
	
	#ifdef USING_HASHVEC
	//compute heldout accuracy when using hash
	double calcAcc(pair<int, Float>** v, int* size_v, int* hashindices, Float lambda){
		hit=0.0;
		margin_hit = 0.0;
		for(int i=0;i<heldout->N;i++){
			memset(prod, 0.0, sizeof(Float)*K);

			SparseVec* xi = heldout->data.at(i);
			Labels* yi = &(heldout->labels.at(i));
			int top = T;
			if (top == -1)
				top = yi->size();
			// compute <w_k, x_i> where w_k is stored in hashmap
			for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
				int j= it->first;
				Float xij = it->second;
				pair<int, Float>* vj = v[j];
				int size_vj0 = size_v[j] - 1;
				for (int k = 0; k < K; k++){
					int index_v = 0;
					find_index(vj, index_v, k, size_vj0, hashindices);
					Float wjk = prox_l1(vj[index_v].second, lambda);
					prod[k] += wjk * xij;
				}
			}
			
			//sort to get rank
			sort(max_indices, max_indices+K, ScoreComp(prod));
			
			for(int k=0;k<top;k++){
				bool flag = false;
				for (int j = 0; j < yi->size(); j++){
					if (yi->at(j) == max_indices[k] ){
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
	//compute heldout accuracy without using hash
	double calcAcc(pair<Float, Float>** v){
		hit=0.0;
		margin_hit = 0.0;
		for(int i=0;i<N;i++){
			memset(prod, 0.0, sizeof(Float)*K);
				
			SparseVec* xi = heldout->data.at(i);
			Labels* yi = &(heldout->labels.at(i));
			
			int top = T;
			if (top == -1)
				top = yi->size();

			// compute <w_k, x_i>
			for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){

				int j= it->first;
				Float xij = it->second;
				pair<Float, Float>* vj = v[j];
				for (int k = 0; k < K; k++){	
					prod[k] += vj[k].second * xij;
				}
			}

			// sort to get rank
			sort(max_indices, max_indices+K, ScoreComp(prod));
			
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
	double calcAcc(pair<int, pair<Float, Float>>** v, int* size_v, vector<int>**& nnz_index, int*& hashindices, int split_up_rate){
	#else
	double calcAcc(pair<Float, Float>** v, vector<int>**& nnz_index, int split_up_rate){
	#endif
		vector<SparseVec*>* data = &(heldout->data);
		vector<Labels>* labels = &(heldout->labels);
		hit=0.0;
		margin_hit = 0.0;
		for(int i=0;i<heldout->N;i++){
			memset(prod, 0.0, sizeof(Float)*K);
			
			SparseVec* xi = data->at(i);
			Labels* yi = &(labels->at(i));
			
			int top = T;
			if (top == -1)
				top = yi->size();
			for (int tt = 0; tt < top; tt++)
				max_indices[tt] = -1;
			for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){

				int j= it->first;
				Float xij = it->second;
				#ifdef USING_HASHVEC
				pair<int, pair<Float, Float>>* vj = v[j];
				int size_vj0 = size_v[j] - 1;
				#else
				pair<Float, Float>* vj = v[j];
				#endif
				for (int S = 0; S < split_up_rate; S++){
					vector<int>& wjS = nnz_index[j][S];
					for (vector<int>::iterator it2 = wjS.begin(); it2 != wjS.end(); it2++){
						int k = *it2;
						#ifdef USING_HASHVEC
						int index_v = 0;
						find_index(vj, index_v, k, size_vj0, hashindices);
						Float wjk = vj[index_v].second.second;
						#else
						Float wjk = vj[k].second;
						#endif
						if (wjk == 0.0 || inside[k]){
							*it2=*(wjS.end()-1);
							wjS.erase(wjS.end()-1);
							it2--;
							continue; 
						}
						inside[k] = true;
						prod[k] += wjk * xij;
						update_max_indices(max_indices, prod, k, top);
					}
					for (vector<int>::iterator it2 = wjS.begin(); it2 != wjS.end(); it2++){
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
	Float* prod;
	int* max_indices;
	Float hit, margin_hit;
	bool* inside;
	int T;
};
class Param{
	public:
	char* trainFname;
	char* modelFname;
	char* heldoutFname;
	Float lambda; //for L1-norm (default 1/N)
	Float C; //weight of loss
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
	int early_terminate;
	
	Param(){
		solver = 1;
		lambda = 0.1;
		C = 1.0;
		max_iter = 50;
		max_select = -1;
		speed_up_rate = -1;
		split_up_rate = 1;
		using_importance_sampling = true;
		post_solve_iter = INF;
		early_terminate = 3;
		heldoutFname == NULL;
		train = NULL;
	}

	~Param(){
		delete[] trainFname;
		delete[] modelFname;
		delete[] heldoutFname;
	}
};

//only used for training
class Model{
	public:
	Model(){
		label_name_list = new vector<string>();
		label_index_map = new map<string,int>();
	}

	#ifdef USING_HASHVEC
	Model(Problem* prob, vector<int>* _w_hash_nnz_index, pair<int, Float>** _w, int* _size_w, int* _hashindices){
		D = prob->D;
		K = prob->K;
		label_name_list = &(prob->label_name_list);
		label_index_map = &(prob->label_index_map);
		w_hash_nnz_index = _w_hash_nnz_index;	
		w = _w;
		size_w = _size_w;
                hashindices = _hashindices;
	}
	pair<int, Float>** w;
	int* size_w;
	int* hashindices;
	#else
	Model(Problem* prob, vector<int>* _w_hash_nnz_index, Float** _w){
		D = prob->D;
		K = prob->K;
		label_name_list = &(prob->label_name_list);
		label_index_map = &(prob->label_index_map);
		w_hash_nnz_index = _w_hash_nnz_index;
		w = _w;
	}
	Float** w;
	#endif
	
	//write model to file
	void writeModel( char* fname){

		ofstream fout(fname);
		fout << "nr_class " << K << endl;
		fout << "label ";
		for(vector<string>::iterator it=label_name_list->begin();
				it!=label_name_list->end(); it++){
			fout << *it << " ";
		}
		fout << endl;
		fout << "nr_feature " << D << endl;
		for(int j=0;j<D;j++){
			vector<int>* nnz_index_j = &(w_hash_nnz_index[j]);
			fout << nnz_index_j->size() << " ";
#ifdef USING_HASHVEC
			pair<int, Float>* wj = w[j];
			int size_wj = size_w[j];
			int size_wj0 = size_wj-1;
			for (vector<int>::iterator it = nnz_index_j->begin(); it != nnz_index_j->end(); it++){
				int k = *it;
				int index_w = 0;
				find_index(wj, index_w, k, size_wj0, hashindices);
				fout << k << ":" << wj[index_w].second << " ";
			}
#else
			Float* wj = w[j];
			for(vector<int>::iterator it=nnz_index_j->begin(); it!=nnz_index_j->end(); it++){
				fout << *it << ":" << wj[*it] << " ";
			}
#endif
			fout << endl;
		}
		fout.close();
		
	}

	HashVec** Hw;
	vector<int>* w_hash_nnz_index;	
	int D;
	int K;
	vector<string>* label_name_list;
	map<string,int>* label_index_map;
};

//only used for prediction
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
		size_t found = line_str.find("  ");
		while (found != string::npos){
			line_str = line_str.replace(found, 2, " ");
			found = line_str.find("  ");
		}
		found = line_str.find(", ");
		while (found != string::npos){
			line_str = line_str.replace(found, 2, ",");
			found = line_str.find(", ");
		}
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
			vector<string> subtokens = split(tokens[st], ",");
			for (vector<string>::iterator it_str = subtokens.begin(); it_str != subtokens.end(); it_str++){
				string str = *it_str;
				if (str == "" || str == " ")
					continue;
				if( (it=label_index_map->find(str)) == label_index_map->end() ){
					lab_indices.push_back(label_index_map->size());
					label_index_map->insert(make_pair(str, lab_indices.back()));
				}else{
					lab_indices.push_back(it->second);
				}
			}
			st++;
		}
		
		SparseVec* ins = new SparseVec();
		for(int i=st;i<tokens.size();i++){
			vector<string> kv = split(tokens[i],":");
			int ind = atoi(kv[0].c_str());
			Float val = atof(kv[1].c_str());
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
