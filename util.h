#ifndef UTIL
#define UTIL

#include<cmath>
#include<vector>
#include<map>
#include<string>
#include<cstring>
#include<stdlib.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <unordered_map>
#include <time.h>
using namespace std;

typedef vector<pair<int,double> > SparseVec;
typedef unordered_map<int,double> HashVec;
typedef vector<int> Labels;
typedef float float_type;
const int LINE_LEN = 100000000;
const int FNAME_LEN = 1000;

#define INFI 1e10
#define INIT_SIZE 16
#define PermutationHash HashClass

class ScoreComp{
	
	public:
	ScoreComp(float_type* _score){
		score = _score;
	}
	bool operator()(const int& ind1, const int& ind2){
		return score[ind1] > score[ind2];
	}
	private:
	float_type* score;
};

// Hash function [K] ->[m]

class HashFunc{
	
	public:
	int* hashindices;
	HashFunc(){
	}
	HashFunc(int _K){
		srand(time(NULL));
		K = _K;
		l = 10000;
		r = 100000;
		
		// pick random prime number in [l, r]
		p = rand() % (r - l) + l - 1;
		bool isprime;
		do {
			p++;
			isprime = true;
			for (int i = 2; i * i <= p; i++){
				if (p % i == 0){
					isprime = false;
					break;
				}
			}
		} while (!isprime);
		a = rand() % p;
		b = rand() % p;
		c = rand() % p;
		hashindices = new int[K];
		for (int i = 0; i < K; i++){
			hashindices[i] = ((a*i*i + b*i + c) % p) % INIT_SIZE;
			if (i < INIT_SIZE) cerr << hashindices[i] % INIT_SIZE << " ";
		}
		cerr << endl;
	}
	~HashFunc(){
		delete [] hashindices;
	}
	void rehash(){
		p = rand() % (r - l) + l - 1;
                bool isprime;
                do {
                        p++;
                        isprime = true;
                        for (int i = 2; i * i <= p; i++){
                                if (p % i == 0){
                                        isprime = false;
                                        break;
                                }
                        }
                } while (!isprime);
		a = rand() % p;
                b = rand() % p;
		for (int i = 0; i < K; i++){
                        hashindices[i] = (a * i + b) % p;
                }
	}
	private:
	int K, l, r;
	int a,b,c,p;
};

class PermutationHash{
	public:
	PermutationHash(){};
	PermutationHash(int _K){	
		srand(time(NULL));
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
