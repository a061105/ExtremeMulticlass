#ifndef NEWHASH
#define NEWHASH

	inline void find_index(pair<int, pair<float_type, float_type> >*& l, int& st, const int& index, const int& size0, int*& hashindices){
	        st = hashindices[index] & size0;
	        //cout << "start finding " << st << " " << size0 << endl;
	        if (l[st].first != index){
	                while (l[st].first != index && l[st].first != -1){
	                        st++;
	                        st &= size0;
	                }
	        }
	        //cout << "end finding " << endl;
	}
	inline void find_index(pair<int, float_type>*& l, int& st, const int& index, const int& size0, int*& hashindices){
	        st = hashindices[index] & size0;
	        //cout << "start finding " << st << " " << size0 << endl;
	        if (l[st].first != index){
	                while (l[st].first != index && l[st].first != -1){
	                        st++;
	                        st &= size0;
	                }
	        }
	        //cout << "end finding " << endl;
	}
	


	/*inline void find_index(pair<int, float_type>* l, int& st, const int& index, const int& size0){
	        //st = hashindices[index] & size0;
	        //cout << "start finding " << st << " " << size0 << endl;
	        if (l[st].first != index){
	                while (l[st].first != index && l[st].first != -1){
	                        st++;
	                        st &= size0;
	                }
	        }
	        //cout << "end finding " << endl;
	}*/
	inline void resize(pair<int, pair<float_type, float_type> >*& l, pair<int, pair<float_type, float_type> >*& L, int& size, int& new_size, int& new_size0, const int& util, int*& hashindices){
		while (util > UPPER_UTIL_RATE * new_size){
	                new_size *= 2;
	        }
	        //while (new_size > INIT_SIZE && util < LOWER_UTIL_RATE * new_size){
	        //        new_size /= 2;
	        //}
	        if (new_size == size)
	                return;
	        pair<int, pair<float_type, float_type>>* new_l = new pair<int, pair<float_type, float_type> >[new_size];
	        new_size0 = new_size - 1;
	        int index_l = 0;
	        for (int tt = 0; tt < new_size; tt++){
	                new_l[tt] = make_pair(-1, make_pair(0.0, 0.0));
	        }
	        for (int tt = 0; tt < size; tt++){
	                //insert old elements into new array
	                pair<int, pair<float_type, float_type>> p = l[tt];
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
	inline void resize(pair<int, float_type>*& l, pair<int, float_type>*& L, int& size, int& new_size, int& new_size0, const int& util, int*& hashindices){
		while (util > UPPER_UTIL_RATE * new_size){
	                new_size *= 2;
	        }
	        //while (new_size > INIT_SIZE && util < LOWER_UTIL_RATE * new_size){
	        //        new_size /= 2;
	        //}
	        if (new_size == size)
	                return;
	        pair<int, float_type>* new_l = new pair<int, float_type>[new_size];
	        new_size0 = new_size - 1;
	        int index_l = 0;
	        for (int tt = 0; tt < new_size; tt++){
	                new_l[tt] = make_pair(-1, 0.0);
	        }
	        for (int tt = 0; tt < size; tt++){
	                //insert old elements into new array
	                pair<int, float_type> p = l[tt];
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

	inline void trim(pair<int, float_type>*& l, int& size, int& util, int*& hashindices){
		util = 0;
		for (int tt = 0; tt < size; tt++){
			pair<int, float_type> p = l[tt];
			if (p.first != -1 && p.second != 0.0)
				util++;
			//else 
			//	l[tt] = make_pair(-1, 0.0);
		}
		int new_size = size;
		while (util > UPPER_UTIL_RATE * new_size){
	                new_size *= 2;
	        }
	        while (new_size > INIT_SIZE && util < LOWER_UTIL_RATE * new_size){
	                new_size /= 2;
	        }
	        pair<int, float_type>* new_l = new pair<int, float_type>[new_size];
	        int new_size0 = new_size - 1;
	        int index_l = 0;
	        for (int tt = 0; tt < new_size; tt++){
	                new_l[tt] = make_pair(-1, 0.0);
	        }
	        for (int tt = 0; tt < size; tt++){
	                //insert old elements into new array
	                pair<int, float_type> p = l[tt];
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
