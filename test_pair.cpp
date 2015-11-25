#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace std;

int main(){
	pair<int, double>* a = new pair<int, double>[100];
	memset(a, 255, sizeof(pair<int, double>)*100);
	a[10].second = 0.0;
	cout << a[10].first << " " << a[10].second << endl;
	int * b = new int [100];
	memset(b, 255, sizeof(b));
	cout << b[1] << " " << sizeof(b) << " " << sizeof(pair<int, double>) << endl;
}
