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
			Float val = atof(ind_val[1].c_str());
			model->w[j][r].first = k;
			model->w[j][r].second = val;
		}
	}
	
	delete[] tmp;
	return model;
}

int main(int argc, char** argv){
	
	if( argc < 1+2 ){
		cerr << "multiPred [testfile] [model] (k) (compute top k accuracy, default 1)" << endl;
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
	Float hit=0.0;
	Float margin_hit = 0.0;
	Float* prod = new Float[model->K];
	int* max_indices = new int[model->K+1];
	for(int k = 0; k < model->K+1; k++){
		max_indices[k] = -1;
	}
	for(int i=0;i<prob->N;i++){
		memset(prod, 0.0, sizeof(Float)*model->K);
		
		SparseVec* xi = data->at(i);
		Labels* yi = &(labels->at(i));
		int top = T;
		if (top == -1)
			top = yi->size();
		for(int ind = 0; ind < top+1; ind++){
			max_indices[ind] = -1;
		}
		if (top == 1)
			max_indices[0] = 0;
		for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
			
			int j= it->first;
			Float xij = it->second;
			if( j >= model->D )
				continue;
			SparseVec* wj = &(model->w[j]);
			for(SparseVec::iterator it2=wj->begin(); it2!=wj->end(); it2++){
				int k = it2->first;
				prod[k] += it2->second*xij;
				update_max_indices(max_indices, prod, k, top);
			}
		}
		Float max_val = -1e300;
		int argmax;
		if (max_indices[0] == -1 || prod[max_indices[0]] < 0.0){
			for (int t = 0; t < top; t++){
				for (int k = 0; k < model->K; k++){
					if (prod[k] == 0.0){
						if (update_max_indices(max_indices, prod, k, top)){
							break;
						}
					}
				}
			}
		}
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
	cerr << "Acc=" << ((Float)hit/prob->N) << endl;
	cerr << "pred time=" << (end-start) << " s" << endl;
}
