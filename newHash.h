#ifndef NEWHASH
#define NEWHASH

	inline void find_index(pair<int, pair<Float, Float> >*& l, int& st, const int& index, const int& size0, int*& hashindices){
	        st = hashindices[index] & size0;
	        if (l[st].first != index){
	                while (l[st].first != index && l[st].first != -1){
	                        st++;
	                        st &= size0;
	                }
	        }
	}
	inline void find_index(pair<int, Float>*& l, int& st, const int& index, const int& size0, int*& hashindices){
	        st = hashindices[index] & size0;
	        if (l[st].first != index){
	                while (l[st].first != index && l[st].first != -1){
	                        st++;
	                        st &= size0;
	                }
	        }
	}
	


	inline void resize(pair<int, pair<Float, Float> >*& l, pair<int, pair<Float, Float> >*& L, int& size, int& new_size, int& new_size0, const int& util, int*& hashindices){
		while (util > UPPER_UTIL_RATE * new_size){
	                new_size *= 2;
	        }
	        if (new_size == size)
	                return;
	        pair<int, pair<Float, Float>>* new_l = new pair<int, pair<Float, Float> >[new_size];
	        new_size0 = new_size - 1;
	        int index_l = 0;
	        for (int tt = 0; tt < new_size; tt++){
	                new_l[tt] = make_pair(-1, make_pair(0.0, 0.0));
	        }
	        for (int tt = 0; tt < size; tt++){
	                //insert old elements into new array
	                pair<int, pair<Float, Float>> p = l[tt];
	                if (p.first != -1 && p.second.first != 0.0){
	                        find_index(new_l, index_l, p.first, new_size0, hashindices);
	                        new_l[index_l] = p;
	                }
	        }
	        delete[] l;
	        l = new_l;
	        size = new_size;
	        L = new_l;
	}
	inline void resize(pair<int, Float>*& l, pair<int, Float>*& L, int& size, int& new_size, int& new_size0, const int& util, int*& hashindices){
		while (util > UPPER_UTIL_RATE * new_size){
	                new_size *= 2;
	        }
	        if (new_size == size)
	                return;
	        pair<int, Float>* new_l = new pair<int, Float>[new_size];
	        new_size0 = new_size - 1;
	        int index_l = 0;
	        for (int tt = 0; tt < new_size; tt++){
	                new_l[tt] = make_pair(-1, 0.0);
	        }
	        for (int tt = 0; tt < size; tt++){
	                //insert old elements into new array
	                pair<int, Float> p = l[tt];
	                if (p.first != -1 && p.second != 0.0){
	                        find_index(new_l, index_l, p.first, new_size0, hashindices);
	                        new_l[index_l] = p;
	                }
	        }
	        delete[] l;
	        l = new_l;
	        size = new_size;
	        L = new_l;
	}

	inline void trim(pair<int, Float>*& l, int& size, int& util, int*& hashindices){
		util = 0;
		for (int tt = 0; tt < size; tt++){
			pair<int, Float> p = l[tt];
			if (p.first != -1 && p.second != 0.0)
				util++;
		}
		int new_size = size;
		while (util > UPPER_UTIL_RATE * new_size){
	                new_size *= 2;
	        }
	        while (new_size > INIT_SIZE && util < LOWER_UTIL_RATE * new_size){
	                new_size /= 2;
	        }
	        pair<int, Float>* new_l = new pair<int, Float>[new_size];
	        int new_size0 = new_size - 1;
	        int index_l = 0;
	        for (int tt = 0; tt < new_size; tt++){
	                new_l[tt] = make_pair(-1, 0.0);
	        }
	        for (int tt = 0; tt < size; tt++){
	                //insert old elements into new array
	                pair<int, Float> p = l[tt];
	                if (p.first != -1 && p.second != 0.0){
	                        find_index(new_l, index_l, p.first, new_size0, hashindices);
	                        new_l[index_l] = p;
	                }
	        }
	        delete[] l;
	        l = new_l;
	        size = new_size;
	}
#endif
