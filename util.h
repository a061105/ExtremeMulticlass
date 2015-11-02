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
using namespace std;

typedef vector<pair<int,double> > SparseVec;
const int LINE_LEN = 100000000;
const int FNAME_LEN = 1000;

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


#endif
