#ifndef UTIL
#define UTIL

#include<cmath>
#include<vector>
#include<map>
#include<string>
#include<cstring>
#include<stdlib.h>
#include<fstream>
#include<iostream>
#include<algorithm>
#include<omp.h>
#include<unordered_map>
#include<time.h>
#include<tuple>
#include<cassert>
#include<limits.h>
using namespace std;

typedef vector<pair<int,double> > SparseVec;
typedef unordered_map<int,double> HashVec;
typedef vector<int> Labels;
typedef double Float;
const int LINE_LEN = 100000000;
const int FNAME_LEN = 1000;
const int INF = INT_MAX;

#define EPS 1e-12
#define INFI 1e10
#define INIT_SIZE 16
#define PermutationHash HashClass
#define UPPER_UTIL_RATE 0.75
#define LOWER_UTIL_RATE 0.5

class ScoreComp{
	
	public:
	ScoreComp(Float* _score){
		score = _score;
	}
	bool operator()(const int& ind1, const int& ind2){
		return score[ind1] > score[ind2];
	}
	private:
	Float* score;
};

class PermutationHash{
	public:
	PermutationHash(){};
	PermutationHash(int _K){	
		K = _K;
		hashindices = new int[K];
		for (int i = 0; i < K; i++){
			hashindices[i] = i;
		}
		random_shuffle(hashindices, hashindices+K);
	}
	~PermutationHash(){
		delete [] hashindices;
	}
	int* hashindices;
	private:
	int K;
};

vector<string> split(string str, string pattern){

	vector<string> str_split;
	size_t i=0;
	size_t index=0;
	while( index != string::npos ){

		index = str.find(pattern,i);
		str_split.push_back(str.substr(i,index-i));

		i = index+1;
	}
	
	if( str_split.back()=="" )
		str_split.pop_back();

	return str_split;
}

double inner_prod(double* w, SparseVec* sv){

	double sum = 0.0;
	for(SparseVec::iterator it=sv->begin(); it!=sv->end(); it++)
		sum += w[it->first]*it->second;
	return sum;
}

double prox_l1_nneg( double v, double lambda ){
	
	if( v < lambda )
		return 0.0;

	return v-lambda;
}

inline Float prox_l1( Float v, Float lambda ){
	
	if( fabs(v) > lambda ){
		if( v>0.0 )
			return v - lambda;
		else 
			return v + lambda;
	}
	
	return 0.0;
}

double norm_sq( double* v, int size ){

	double sum = 0.0;
	for(int i=0;i<size;i++){
		if( v[i] != 0.0 )
			sum += v[i]*v[i];
	}
	return sum;
}

int total_size( vector<int>* alpha, int size ){
	
	int sum = 0;
	for(int i=0;i<size;i++)
		sum += alpha[i].size();
	return sum;
}

int total_size( HashVec** w, int size ){
	
	int sum = 0;
	for(int j=0;j<size;j++)
		sum += w[j]->size();
	return sum;
}

long nnz( vector<SparseVec*>& data ){
	
	long sum =0;
	for(int i=0;i<data.size();i++){
		sum += data[i]->size();
	}
	return sum;
}

// maintain top tK indices, stored in max_indices, where indices are sorted by x[].
// Here the situation is x(i) has just been updated, where i may or may not exist in max_indices
inline bool update_max_indices(int* max_indices, Float* x, int i, int tK){
	//max_indices should have size tK+1
	int ind = 0;
	// entry ind is empty if max_indices[ind] == -1
	while (ind < tK && max_indices[ind] != -1 && max_indices[ind] != i){
		ind++;
	}
	if (ind < tK && max_indices[ind] == i)
		return false;
	max_indices[ind] = i;
	int k = 0;
	//try move to right
	while (ind < tK-1 && max_indices[ind+1] != -1 && x[max_indices[ind+1]] > x[max_indices[ind]]){
                k = max_indices[ind];
                max_indices[ind] = max_indices[ind+1];
                max_indices[++ind] = k;
        }
	//try move to left
	while (ind > 0 && x[max_indices[ind]] > x[max_indices[ind-1]]){
		k = max_indices[ind];
		max_indices[ind] = max_indices[ind-1];
		max_indices[--ind] = k;
	}
	return true;
}

//min_{x,y} \|x - b\|^2 + \|y - c\|^2
// s.t. x >= 0, y >= 0
//  \|x\|_1 = \|y\|_1 = t \in [0, C]
// x,b \in R^n; y,c \in R^m
// O( (m + n) log(m+n) ), but usually dominated by complexity of computing b, c
inline void solve_bi_simplex(int n, int m, Float* b, Float* c, Float C, Float* x, Float* y){
	int* index_b = new int[n];
	int* index_c = new int[m];
	for (int i = 0; i < n; i++)
		index_b[i] = i;
	for (int j = 0; j < m; j++)
		index_c[j] = j;
	sort(index_b, index_b+n, ScoreComp(b));
	sort(index_c, index_c+m, ScoreComp(c));
	Float* S_b = new Float[n];
	Float* S_c = new Float[m];
	Float* D_b = new Float[n+1];
	Float* D_c = new Float[m+1];
	Float r_b = 0.0, r_c = 0.0;
	for (int i = 0; i < n; i++){
		r_b += b[index_b[i]]*b[index_b[i]];
		if (i == 0)
			S_b[i] = b[index_b[i]];
		else
			S_b[i] = S_b[i-1] + b[index_b[i]];
		D_b[i] = S_b[i] - (i+1)*b[index_b[i]];
	}
	D_b[n] = C;
	for (int j = 0; j < m; j++){
		r_c += c[index_c[j]]*c[index_c[j]];
		if (j == 0)
			S_c[j] = c[index_c[j]];
		else
			S_c[j] = S_c[j-1] + c[index_c[j]];
		D_c[j] = S_c[j] - (j+1)*c[index_c[j]];
	}
	D_c[m] = C;
	int i = 0, j = 0;
	//update for b_{0..i-1} c_{0..j-1}
	//i,j is the indices of coordinate that we will going to include, but not now!
	Float t = 0.0;
	Float ans_t_star = 0;
	Float ans = INFI;
	int ansi = i, ansj = j;
	int lasti = 0, lastj = 0;
	do{
		lasti = i; lastj = j;
		Float l = t;
		t = min(D_b[i+1], D_c[j+1]);
		//now allowed to use 0..i, 0..j
		if (l >= C && t > C){
			break;
		}
		if (t > C) { 
			t = C;
		}
		Float t_star = ((i+1)*S_c[j] + (1+j)*S_b[i])/(i+j+2);
		//cerr << "getting t_star=" << t_star << endl;
		if (t_star < l){
			t_star = l;
		//	cerr << "truncating t_star=" << l << endl;
		}
		if (t_star > t){
			t_star = t;
		//	cerr << "truncating t_star=" << t << endl;
		}
		Float candidate = r_b + r_c + (S_b[i] - t_star)*(S_b[i] - t_star)/(i+1) + (S_c[j] - t_star)*(S_c[j] - t_star)/(j+1);
		//cerr << "candidate val=" << candidate << endl;
		if (candidate < ans){
			ans = candidate;
			ansi = i;
			ansj = j;
			ans_t_star = t_star;
		}
		while ((i + 1)< n && D_b[i+1] <= t){
			i++;
			r_b -= b[index_b[i]]*b[index_b[i]];
		}
		//cerr << "updating i to " << i << endl;
		while ((j+1) < m && D_c[j+1] <= t) {
			j++;
			r_c -= c[index_c[j]]*c[index_c[j]];
		}
		//cerr << "updating j to " << j << endl;
	} while (i != lasti || j != lastj);
	//cerr << "ansi=" << ansi << ", ansj=" << ansj << ", t_star=" << ans_t_star << endl;
	for(i = 0; i < n; i++){
		int ii = index_b[i];
		if (i <= ansi)
			x[ii] = (b[index_b[i]] + (ans_t_star - S_b[ansi])/(ansi+1));
		else
			x[ii] = 0.0;
	}
	for(j = 0; j < m; j++){
		int jj = index_c[j];
		if (j <= ansj)
			y[jj] = c[index_c[j]] + (ans_t_star - S_c[ansj])/(ansj+1);
		else
			y[jj] = 0.0;
	}

	delete[] S_b; delete[] S_c;
	delete[] index_b; delete[] index_c;
	delete[] D_b; delete[] D_c;
}

#endif
