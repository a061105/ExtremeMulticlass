#include "multi.h"
#include <omp.h>
#include <cassert>

StaticModel* readModel(char* file){
	
	StaticModel* model = new StaticModel();
	
	ifstream fin(file);
	char* tmp = new char[LINE_LEN];
	fin >> tmp >> (model->K);
	
	fin >> tmp;
	string name;
	for(int k=0;k<model->K;k++){
		fin >> name;
		model->label_name_list->push_back(name);
		model->label_index_map->insert(make_pair(name,k));
	}
	
	fin >> tmp >> (model->D);
	model->w = new SparseVec[model->D];
	
	vector<string> ind_val;
	int nnz_j;
	for(int j=0;j<model->D;j++){
		fin >> nnz_j;
		model->w[j].resize(nnz_j);
		for(int r=0;r<nnz_j;r++){
			fin >> tmp;
			ind_val = split(tmp,":");
			int k = atoi(ind_val[0].c_str());
			float_type val = atof(ind_val[1].c_str());
			model->w[j][r].first = k;
			model->w[j][r].second = val;
		}
	}
	
	delete[] tmp;
	return model;
}

int main(int argc, char** argv){
	
	if( argc < 1+2 ){
		cerr << "multiPred [testfile] [model] (top k, default auto)" << endl;
		exit(0);
	}
	

	char* testFile = argv[1];
	char* modelFile = argv[2];
	int T = 1;
	if (argc > 3){
		T = atoi(argv[3]);
	}
	
	StaticModel* model = readModel(modelFile);
	
	Problem* prob = new Problem();
	readData( testFile, prob );
	
	cerr << "Ntest=" << prob->N << endl;
	
	double start = omp_get_wtime();
	//compute accuracy
	vector<SparseVec*>* data = &(prob->data);
	vector<Labels>* labels = &(prob->labels);
	float_type hit=0.0;
	float_type margin_hit = 0.0;
	float_type* prod = new float_type[model->K];
	int* k_index = new int[model->K];
	for(int k = 0; k < model->K; k++){
		k_index[k] = k;
	}
	int nnz_wj_sum = 0;
	int nnz_x = 0;
	for(int i=0;i<prob->N;i++){
		memset(prod, 0.0, sizeof(float_type)*K);
		//for(int k=0;k<model->K;k++)
		//	prod[k] = 0.0;
		
		SparseVec* xi = data->at(i);
		Labels* yi = &(labels->at(i));
		int top = T;
		if (top == -1)
			top = yi->size();
		int* max_indices = new int[top+1];
		for(int ind = 0; ind <= top; ind++){
			max_indices[ind] = -1;
		}
		if (top == 1)
			max_indices[0] = 0;
		nnz_x += xi->size();
		for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
			
			int j= it->first;
			float_type xij = it->second;
			if( j >= model->D )
				continue;
			SparseVec* wj = &(model->w[j]);
			nnz_wj_sum += wj->size();
			for(SparseVec::iterator it2=wj->begin(); it2!=wj->end(); it2++){
				int k = it2->first;
				prod[k] += it2->second*xij;
				if (top == 1){
					if (prod[max_indices[0]] < prod[k]){
						max_indices[0] = k;
					}
					continue;
				}
				int ind = 0;
				while (ind < top && max_indices[ind] != -1 && max_indices[ind] != k){
					ind++;
				}
				max_indices[ind] = k;
				//try move to left
				while (ind > 0 && prod[max_indices[ind]] > prod[max_indices[ind-1]]){
					//using k as temporary variables
					k = max_indices[ind];
					max_indices[ind] = max_indices[ind-1];
					max_indices[ind-1] = k;
				}
				//try move to right
				while (ind < top-1 && max_indices[ind+1] != -1 && prod[max_indices[ind+1]] > prod[max_indices[ind]]){
                                        //using k as temporary variables
                                        k = max_indices[ind];
                                        max_indices[ind] = max_indices[ind+1];
                                	max_indices[ind+1] = k;
                                }
			}
		}
		
		//sort(k_index, k_index + model->K, ScoreComp(prod));
		float_type max_val = -1e300;
		int argmax;
		for(int k=0;k<top;k++){
			bool flag = false;
			for (int j = 0; j < yi->size(); j++){
				if (prob->label_name_list[yi->at(j)] == model->label_name_list->at(max_indices[k])){
					flag = true;
				}
			}
			if (flag)
				hit += 1.0/top;
		}
	}
	
	double end = omp_get_wtime();
	cerr << "k_eff=" << (float_type)nnz_wj_sum/nnz_x << endl;
	cerr << "Acc=" << ((float_type)hit/prob->N) << endl;
	cerr << "pred time=" << (end-start) << " s" << endl;
}
