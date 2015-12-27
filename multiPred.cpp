#include "multi.h"
#include <omp.h>

Model* readModel(char* file){
	
	Model* model = new Model();

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
	model->Hw = new HashVec*[model->D];
	
	vector<string> ind_val;
	int nnz_j;
	for(int j=0;j<model->D;j++){
		fin >> nnz_j;
		HashVec* wj = new HashVec(nnz_j);
		for(int r=0;r<nnz_j;r++){
			fin >> tmp;
			ind_val = split(tmp,":");
			int k = atoi(ind_val[0].c_str());
			float_type val = atof(ind_val[1].c_str());
			wj->insert(make_pair(k,val));
		}
		model->Hw[j] = wj;
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
	int T = -1;
	if (argc > 3){
		T = atoi(argv[3]);
	}
	
	Model* model;
	model = readModel(modelFile);
	
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
	for(int i=0;i<prob->N;i++){
		for(int k=0;k<model->K;k++)
			prod[k] = 0.0;
		
		SparseVec* xi = data->at(i);
		Labels* yi = &(labels->at(i));
		int top = T;
		if (top == -1)
			top = yi->size();
		for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
			
			int j= it->first;
			float_type xij = it->second;
			if( j >= model->D )
				continue;
			
			HashVec* wj = model->Hw[j];
			for(HashVec::iterator it2=wj->begin(); it2!=wj->end(); it2++){
				prod[it2->first] += it2->second*xij;
			}
		}
		sort(k_index, k_index + model->K, ScoreComp(prod));
		float_type max_val = -1e300;
		int argmax;
		for(int k=0;k<top;k++){
			bool flag = false;
			for (int j = 0; j < yi->size(); j++){
				if (prob->label_name_list[yi->at(j)] == model->label_name_list->at(k_index[k])){
					flag = true;
				}	
			}
			if (flag)
				hit += 1.0/top;
		}
	}
	double end = omp_get_wtime();

	cerr << "Acc=" << ((float_type)hit/prob->N) << endl;
	cerr << "pred time=" << (end-start) << " s" << endl;
}
