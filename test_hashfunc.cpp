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

int main(){
	double t1 = 0.0, t2 = 0.0;
	int a = rand(), b = rand(), p = rand(), m = rand();
	t1 -= omp_get_wtime();
	double ans = 0.0;
	for(int tt = 0; tt < 10000000; tt++){
		for(int i = 0; i < 10000; i++){
			int j = ((a*i+b) % p) % m + tt;
			ans += j;
		}
	}
	t1 += omp_get_wtime();
	int* hashfunc = new int[10000000];
	for(int i = 0; i < 10000; i++){
                hashfunc[i] = ((a*i+b) % p) % m;
        }
	t2 -= omp_get_wtime();
        for(int tt = 0; tt < 10000000; tt++){
                for(int i = 0; i < 10000; i++){
                        int j = hashfunc[i] + tt;
			ans += j;
                }
	}
        t2 += omp_get_wtime();
	cout << t1 << " " << t2 << endl;
	return 0;
}
