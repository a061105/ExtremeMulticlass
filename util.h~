#ifndef UTIL
#define UTIL

#include<cmath>
#include<vector>
#include <map>
#include<string>
#include<cstring>
#include<stdlib.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <unordered_map>
using namespace std;

typedef vector<pair<int,double> > SparseVec;
typedef unordered_map<int,double> HashVec;
const int LINE_LEN = 100000000;
const int FNAME_LEN = 1000;

class ScoreComp{
	
	public:
	ScoreComp(double* _score){
		score = _score;
	}
	bool operator()(const int& ind1, const int& ind2){
		return score[ind1] > score[ind2];
	}
	private:
	double* score;
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

double prox_l1( double v, double lambda ){
	
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

#endif
