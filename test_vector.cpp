#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main(){
	vector<int>* a = new vector<int>();
	a->reserve(10000000);
	int* b = new int[10000000];
	double t1 = -omp_get_wtime();
	for(int i = 0; i < 10000000; i++){
		a->push_back(i);
	}
	t1 += omp_get_wtime();
	double t2 = -omp_get_wtime();
        for(int i = 0; i < 10000000; i++){
                b[i] = i;
        }       
        t2 += omp_get_wtime();
	cout << t1 << " " << t2 << endl;
}	
