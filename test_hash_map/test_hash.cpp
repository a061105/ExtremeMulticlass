#include <unordered_map>
#include <map>
#include <omp.h>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
using namespace std;

typedef vector<pair<int,double> > SparseVec;
typedef unordered_map<int,double> HashVec;
typedef map<int,double> TreeVec;

void sumReduce(SparseVec* list){
	
	sort(list->begin(), list->end());
	
	SparseVec* list2 = new SparseVec();

	int ind = -1;
	double val = 0.0;
	for(SparseVec::iterator it=list->begin();it!=list->end();it++){
		if( ind==it->first ){
			val += it->second;
		}else{
			if( ind != -1 )
				list2->push_back(make_pair(ind,val));
			
			ind = it->first;
			val = it->second;
		}
	}
	if( ind != -1 )
		list2->push_back(make_pair(ind,val));
	
	*list = *list2;

	delete list2;
}

int main(){
	
	double start2 = omp_get_wtime();
	int N2 = 1000000;
	int K2 = 100000;
	double** A = new double*[N2];
	for(int i=0;i<N2;i++){
		A[i] = new double[K2];
		for(int k=0;k<K2;k++)
			A[i][k] = 0.0;
	}
	cerr << "time=" << omp_get_wtime()-start2 << endl;
	exit(0);
	//insert (assign) a lot of key-value pair
	int N = 10000;
	int K = 3000;
	int D = 10000;
	int nnz_x_i = 100;
	int nnz_w_j = 100;
	int act_k_size = 100;
	vector<int>* act_k_index = new vector<int>[N];
	for(int i=0;i<N;i++){
		for(int j=0;j<act_k_size;j++){
			act_k_index[i].push_back(rand()%K);
		}
		sort(act_k_index[i].begin(), act_k_index[i].end());
	}
	
	//gen data
	vector<SparseVec*> data;
	for(int i=0;i<N;i++){
		SparseVec* sv = new SparseVec();
		for(int j=0;j<nnz_x_i;j++){
			int ind = rand()%D;
			double val = ((double) rand()/RAND_MAX);
			sv->push_back(make_pair(ind,val));
		}
		sort(sv->begin(), sv->end());
		data.push_back(sv);
	}
	
	//test array update
	double* w = new double[D*K];
	double start = omp_get_wtime();
	for(int i=0;i<N;i++){

		SparseVec* x_i = data[i];
		for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
			int f_ind = it->first;
			int f_val = it->second;
			for(int j=0;j<act_k_size;j++){
				int k = act_k_index[i][j];
				w[f_ind*K+k] += f_val;
			}
		}
	}
	cerr << "array update time=" << omp_get_wtime()-start  << endl;
	
	//test double array update
	double** w2 = new double*[D];
	for(int j=0;j<D;j++)
		w2[j] = new double[K];
	start = omp_get_wtime();
	for(int i=0;i<N;i++){

		SparseVec* x_i = data[i];
		for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
			int f_ind = it->first;
			int f_val = it->second;
			double* w2_j = w2[f_ind];
			for(int j=0;j<act_k_size;j++){
				int k = act_k_index[i][j];
				w2_j[k] += f_val;
			}
		}
	}
	cerr << "two-dimensional array update time=" << omp_get_wtime()-start  << endl;
	
	//test SparseVec update time
	SparseVec** w3 = new SparseVec*[D];
	for(int j=0;j<D;j++){
		w3[j] = new SparseVec();
		w3[j]->reserve(K);
	}
	start = omp_get_wtime();
	for(int i=0;i<N;i++){
		SparseVec* x_i = data[i];
		for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
			int j = it->first;
			double x_ij = it->second;
			SparseVec* w3_j = w3[j];
			for(int r=0;r<act_k_size;r++){
				int k = act_k_index[i][r];
				w3_j->push_back(make_pair(k,x_ij));
			}
		}
	}
	for(int j=0;j<D;j++)
		sumReduce(w3[j]);
	cerr << "sparse vect update time=" << omp_get_wtime()-start << endl;

	//test hash map update
	start = omp_get_wtime();
	HashVec** v = new HashVec*[D];
	for(int j=0;j<D;j++)
		v[j] = new HashVec();
	HashVec::iterator it2;
	for(int i=0;i<N;i++){
		SparseVec* x_i = data[i];
		for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
			int f_ind = it->first;
			int f_val = it->second;
			HashVec* v_j = v[f_ind];
			for(int j=0;j<act_k_size;j++){
				int k = act_k_index[i][j];
				if( (it2=v_j->find(k)) != v_j->end() ){
					it2->second += f_val;
				}else{
					v_j->insert(make_pair(k,f_val));
				}
			}
		}
	}
	cerr << "hashmap update time=" << omp_get_wtime()-start << endl;
	
	

	//test hash map matrix multiplication
	/*
	start = omp_get_wtime();
	double* A = new double[N*K];
	for(int j=0;j<D;j++){
		SparseVec* x_i = data[i];
		for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
			int f_ind = it->first;
			int f_val = it->second;
			
		}
	}*/
}
