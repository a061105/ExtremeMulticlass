#include "util.h"

class A{
	
	public:
	static map<string,int> label_map;
	static int K;
};

map<string,int> A::label_map;
int A::K = 0;

int main(){
	
	A* a = new A();
	a->label_map.insert(make_pair("asdf",1));
	a->K = 1;

	A* a2 = new A();
	cerr << a2->label_map.size() << endl;
	cerr << a2->K << endl;
}
