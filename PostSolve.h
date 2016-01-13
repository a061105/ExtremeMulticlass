#include "util.h"
#include "multi.h"

class PostSolve{
	
	public:
	#ifdef USING_HASHVEC
	PostSolve(Param* param, vector<int>** _w_hash_nnz_index, pair<int, float_type>** _w, int* _size_w, vector<pair<int, float_type>>* _act_k_index, pair<int, pair<float_type, float_type>>** _v, int*& _size_v, int*& _hashindices){
	#else
	PostSolve(Param* param, vector<int>** _w_hash_nnz_index, float_type** _w, vector<pair<int, float_type>>* _act_k_index, pair<float_type, float_type>** _v){
	#endif
		
		double construct_time = -omp_get_wtime();
		
		prob = param->train;
		C = param->C;
		
		vector<SparseVec*>* data = &(prob->data);
		labels = &(prob->labels);
		D = prob->D;
		N = prob->N;
		K = prob->K;
		max_iter = param->post_solve_iter;
		int maxDK = D;
		if (maxDK < K) 
			maxDK = K;
		hashfunc = new HashClass(maxDK);
		hashindices = hashfunc->hashindices;
		
		//initialize alpha and v
		act_k_index = new vector<pair<int, float_type>>[N];
		for(int i = 0; i < N; i++){
			act_k_index[i] = _act_k_index[i];
		}
		#ifdef USING_HASHVEC
		v = new pair<int, float_type>*[K];
		size_v = new int[K];
		util_v = new int[K];
		for(int k = 0; k < K; k++){
			size_v[k] = 1;
			util_v[k] = 0;
		}
		for(int j = 0; j < D; j++){
			for (int it = 0; it < _size_v[j]; it++){
				int k = _v[j][it].first;
				if (k == -1)
					continue;
				if (fabs(_v[j][it].second.second) <= 1e-12)
					continue;
				util_v[k]++;
				if (size_v[k]*UPPER_UTIL_RATE < util_v[k])
					size_v[k]*= 2;
			}
		}
		for(int k = 0; k < K; k++){
			v[k] = new pair<int, float_type>[size_v[k]];
			for (int j = 0; j < size_v[k]; j++)
				v[k][j] = make_pair(-1, 0.0);
		}
		for (int j = 0; j < D; j++){
			for (int it = 0; it < _size_v[j]; it++){
				int k = _v[j][it].first;
				if (k == -1)
					continue;
				if (fabs(_v[j][it].second.second) <= 1e-12)
					continue;
				//insert (j, _v[j][it]) to v[k]
				int index_v = 0;
				find_index(v[k], index_v, j, size_v[k]-1, hashindices);
				v[k][index_v] = make_pair(j, _v[j][it].second.first);
			}
		}
		#else
		v = new float_type*[K]; //w = prox(v);
		for(int k=0;k<K;k++){
			v[k] = new float_type[D];
			for(int j=0;j<D;j++)
				v[k][j] = 0.0;
		}
		for(int j=0;j<D;j++)
			for(vector<int>::iterator it=_w_hash_nnz_index[j]->begin(); it!=_w_hash_nnz_index[j]->end(); it++){
				int k = *it;
				v[k][j] = _v[j][k].first;
			}
		#endif

		// construct data_per_class
		data_per_class = new vector<SparseVec*>[N];
		//for(int i=0;i<N;i++)
		//	data_per_class[i] = new vector<SparseVec*>();

		/*double nnz_alpha_avg = (double)total_size( act_k_index, N ) / N;
		double nnz_w_avg = 0.0;
        	for(int j=0;j<D;j++)
        	        nnz_w_avg += _w_hash_nnz_index[j]->size();
		nnz_w_avg /= D;
		//double nnz_w_avg = (double)total_size( *_w_hash_nnz_index, D ) / D;
		cerr << "nnz_alpha=" << nnz_alpha_avg << ", nnz_w_avg=" << nnz_w_avg << endl;
		*/
		//if( nnz_alpha_avg < nnz_w_avg ){
			
			#ifdef USING_HASHVEC
			for(int i=0;i<N;i++){
				SparseVec* xi = data->at(i);
				vector<SparseVec*>* data_per_class_i = &(data_per_class[i]);
				vector<pair<int, float_type>>* act_k_i = &(act_k_index[i]);
				for(vector<pair<int, float_type>>::iterator it=act_k_i->begin(); it!=act_k_i->end(); it++){
					int k = it->first;
					SparseVec* data_per_class_i_k = new SparseVec();
					//loop over j s.t. x[i][j] != 0
					for(SparseVec::iterator it2=xi->begin(); it2!=xi->end(); it2++){
						//need w[j][k] != 0
						int index_w = 0;
						int j = it2->first;
						find_index(_w[j], index_w, k, _size_w[j]-1, _hashindices);
						if( _w[j][index_w].second != 0.0 )
							data_per_class_i_k->push_back(make_pair(j, it2->second));
					}
					data_per_class_i->push_back(data_per_class_i_k);
				}
			}
				
			#else
			for(int i=0;i<N;i++){
				SparseVec* xi = data->at(i);
				//SparseVec* data_per_class_i = data_per_class[i];
				vector<SparseVec*>* data_per_class_i = &(data_per_class[i]);
				vector<pair<int, float_type>>* act_k_i = &(act_k_index[i]);
				for(vector<pair<int, float_type>>::iterator it=act_k_i->begin(); it!=act_k_i->end(); it++){
					int k = it->first;
					SparseVec* data_per_class_i_k = new SparseVec();
					for(SparseVec::iterator it2=xi->begin(); it2!=xi->end(); it2++){
						if( _w[ it2->first ][k] != 0.0 )
							data_per_class_i_k->push_back(make_pair(it2->first, it2->second));
					}
					data_per_class_i->push_back(data_per_class_i_k);
				}
			}
			#endif
		
		//}
		/*else{
			
			for(int i=0;i<N;i++){
				SparseVec* xi = data->at(i);
				vector<SparseVec*>* data_per_class_i = &(data_per_class[i]);
				//SparseVec* data_per_class_i = data_per_class[i];
				//hashmap(<k, data_per_class_i_k>
				for(SparseVec::iterator it=xi->begin() ;it!=xi->end(); it++){
					int j = it->first;
					double xij = it->second;
					SparseVec* data_per_class_i_k = new SparseVec();
					for(vector<int>::iterator it2=_w_hash_nnz_index[j]->begin(); it2!=_w_hash_nnz_index[j]->end(); it2++){
						int k = *it2;
						#ifdef USING_HASHVEC
						int index_alpha = 0;
						int _size_alphai0 = _size_alpha[i] - 1;
						find_index(_alpha[i], index_alpha, k, _size_alphai0, _hashindices);
						if( _alpha[i][index_alpha].second != 0.0 )
							data_per_class_i_k->push_back(make_pair(j, xij));
						#else
						if( _alpha[i][k] != 0.0)
							data_per_class_i_k->push_back(make_pair(j, xij));
						#endif
					}
					data_per_class_i->push_back(data_per_class_i_k);
				}
			}
		}*/
		
		construct_time += omp_get_wtime();
		cerr << "construct_time=" << construct_time << endl;

		//initialize Q_diag (Q=X*X') for the diagonal Hessian of each i-th subproblem
		Q_diag = new float_type[N];
		for(int i=0;i<N;i++){
			SparseVec* ins = data->at(i);
			float_type sq_sum = 0.0;
			for(SparseVec::iterator it=ins->begin(); it!=ins->end(); it++)
				sq_sum += it->second*it->second;
			Q_diag[i] = sq_sum;
		}
	}
	
	~PostSolve(){
		for(int i=0;i<N;i++){
			for (vector<SparseVec*>::iterator it = data_per_class[i].begin(); it != data_per_class[i].end(); it++){
				SparseVec* data_per_class_i_k = *it;
				data_per_class_i_k->clear();
			}
			data_per_class[i].clear();
		}
		delete[] data_per_class;
		delete[] act_k_index;
		
		delete[] Q_diag;
		for(int k=0;k<K;k++)
			delete[] v[k];
		delete[] v;
	}
	
	Model* solve(){
		
		//indexes for permutation of [N]
		int* index = new int[N];
		for(int i=0;i<N;i++)
			index[i] = i;
		//main loop
		double starttime = omp_get_wtime();
		double subsolve_time = 0.0, maintain_time = 0.0;
		float_type* alpha_i_new = new float_type[K];
		int iter = 0;
		while( iter < max_iter ){
			random_shuffle( index, index+N );
			for(int r=0;r<N;r++){
				
				int i = index[r];
				
				if( act_k_index[i].size() < 2 )
					continue;
				
				//solve subproblem
				subsolve_time -= omp_get_wtime();
				subSolve(i, act_k_index[i], alpha_i_new);
				subsolve_time += omp_get_wtime();
				
				//maintain v = X^T\alpha
				maintain_time -= omp_get_wtime();
				vector<SparseVec*>* x_i_per_class = &(data_per_class[i]);
				vector<SparseVec*>::iterator data_it = x_i_per_class->begin();
				for(vector<pair<int, float_type>>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
					
					int k= it->first;
					SparseVec* x_i = *(data_it++);
					float_type alpha_diff = alpha_i_new[k] - it->second;
					#ifdef USING_HASHVEC
					int size_vk0 = size_v[k] - 1;
					int size_vk = size_v[k];
					pair<int, float_type>* vk = v[k];
					for(SparseVec::iterator it2=x_i->begin() ;it2!=x_i->end(); it2++){
						int index_v = 0, j = it2->first;
						find_index(vk, index_v, j, size_vk0, hashindices);
						vk[index_v].second += it2->second*(alpha_diff);
						if (vk[index_v].first == -1){
							vk[index_v].first = j;
							if ((++util_v[k]) > size_v[k]*UPPER_UTIL_RATE){
								resize(vk, v[k], size_vk, size_v[k], size_vk0, util_v[k], hashindices);	
							}
						}
					}
					#else
					float_type* vk = v[k];
					for(SparseVec::iterator it2=x_i->begin() ;it2!=x_i->end(); it2++){
						vk[it2->first] += it2->second*(alpha_diff);
					}
					#endif
				}
				for(vector<pair<int, float_type>>::iterator it = act_k_index[i].begin(); it != act_k_index[i].end(); it++){
					it->second = alpha_i_new[it->first];
				}
				maintain_time += omp_get_wtime();
			}
			
			if( iter % 1 == 0 )
				cerr << "." ;
			
			iter++;
		}
		double endtime = omp_get_wtime();
		cerr << endl;
		
		//convert v into w
		#ifdef USING_HASHVEC
		w = new pair<int, float_type>*[D];
		size_w = new int[D];
		nnz_index = new vector<int>*[D];
		for (int j = 0; j < D; j++){
			size_w[j] = 2;
			w[j] = new pair<int, float_type>[size_w[j]];
			for (int tt = 0; tt < size_w[j]; tt++){
				w[j][tt] = make_pair(-1, 0.0);
			}
			nnz_index[j] = new vector<int>();
		}
		for (int k = 0; k < K; k++){
			for (int it = 0; it < size_v[k]; it++){
				int j = v[k][it].first;
				pair<int, float_type>* wj = w[j];
				int size_wj = size_w[j];
				if (fabs(v[k][it].second) > 1e-12 ){
					int index_w = 0;
					find_index(wj, index_w, k, size_wj - 1, hashindices); 	
					wj[index_w].second = v[k][it].second;
					if (wj[index_w].first == -1){
						wj[index_w].first = k;
						nnz_index[j]->push_back(k);
						if (size_wj* UPPER_UTIL_RATE < nnz_index[j]->size()){
							int util = nnz_index[j]->size();
							int size_wj0 = size_wj - 1;
							resize(wj, w[j], size_wj, size_w[j], size_wj0, util, hashindices);
						}
					}
				}
			}
		}
		#else
		w = new float_type*[D];
		nnz_index = new vector<int>*[D];
		for (int j = 0; j < D; j++){
			w[j] = new float_type[K];
			for (int k = 0; k < K; k++){
				w[j][k] = 0.0;
			}
			nnz_index[j] = new vector<int>();
		}
		for (int k = 0; k < K; k++){
			for (int j = 0; j < D; j++){
				float_type wjk = v[k][j];
				if (fabs(wjk) > 1e-12){
					w[j][k] = wjk;
					nnz_index[j]->push_back(k);
				}
			}
		}	
		
		#endif
		float_type d_obj = 0.0;
		int nSV = 0;
		int nnz_w = 0;
		double w_1norm=0.0;
		for(int j=0;j<D;j++){
			for(vector<int>::iterator it=nnz_index[j]->begin(); it!=nnz_index[j]->end(); it++){
				int k = *it;
				#ifdef USING_HASHVEC
				int index_w = 0;
				find_index(w[j], index_w, k, size_w[j]-1, hashindices);
				float_type wjk = w[j][index_w].second;
				#else
				float_type wjk = w[j][k];
				#endif
				d_obj += wjk*wjk;
				w_1norm += fabs(wjk);
				
			}
			nnz_w += nnz_index[j]->size();
		}
		d_obj/=2.0;
		for(int i=0;i<N;i++){
			Labels* yi = &(labels->at(i));
			for (vector<pair<int, float>>::iterator it = act_k_index[i].begin(); it != act_k_index[i].end(); it++){
				int k = it->first;
				
				if(find(yi->begin(), yi->end(), k) ==yi->end())
					d_obj += it->second;
				if( fabs( it->second ) > 1e-12 )
					nSV++;
			}
		}
		cerr << "dual_obj=" << d_obj << endl;
		cerr << "nSV=" << nSV << " (NK=" << N*K << ")"<< endl;
		cerr << "nnz_w=" << nnz_w << " (DK=" << D*K << ")" << endl;
		cerr << "w_1norm=" << w_1norm << endl;
		cerr << "train time=" << endtime-starttime << endl;
		cerr << "subsolve time=" << subsolve_time << endl;
		cerr << "maintain time=" << maintain_time << endl;
		
		//delete algorithm-specific variables
		delete[] alpha_i_new;
		delete[] index;
		
		#ifdef USING_HASHVEC
		return new Model(prob, nnz_index, w, size_w, hashindices);
		#else
		return new Model(prob, nnz_index, w);
		#endif
	}
	
	void subSolve(int I, vector<pair<int, float_type>>& act_k_index, float_type* alpha_i_new){
		
		Labels* yi = &(labels->at(I));
		int act_k_size = act_k_index.size();
		
		//number of indices in (yi) and (act_set - yi), respectively
		int j = 0, i = 0;
		int m = yi->size(), n = act_k_size - m;
		
		float_type* b = new float_type[n+m];
		float_type* c = new float_type[m+n];
		int* act_index_b = new int[n+m];
		int* act_index_c = new int[m+n];
		vector<SparseVec*>* x_i_per_class = &(data_per_class[I]);
		
		float_type A = Q_diag[I];

		int* index_b = new int[n+m];
		int* index_c = new int[m+n];
		/*for(int k=0;k<m+n;k++){
			int p = act_k_index[k];
			act_k_index
			#ifdef USING_HASHVEC
			int index_alpha = 0;
			find_index(alpha_i, index_alpha, p, size_alphai0, hashindices);
			if( find(yi->begin(), yi->end(), p) == yi->end() ){
				b[i] = 1.0 - A*alpha_i[index_alpha].second;
				index_b[i] = i;
				invert_index_b[i] = k;
				act_index_b[i++] = p;
			}else{
				c[j] = A*alpha_i[index_alpha].second;
				index_c[j] = j;
				invert_index_c[j] = k;
				act_index_c[j++] = p;
			}
			#else
			if( find(yi->begin(), yi->end(), p) == yi->end() ){
				b[i] = 1.0 - A*alpha_i[p];
				index_b[i] = i;
				invert_index_b[i] = k;
				act_index_b[i++] = p;
			}else{
				c[j] = A*alpha_i[p];
				index_c[j] = j;
				invert_index_c[j] = k;
				act_index_c[j++] = p;
			}
			#endif
		}*/
		vector<SparseVec*>::iterator data_it = x_i_per_class->begin();
		for(vector<pair<int, float_type>>::iterator it = act_k_index.begin(); it != act_k_index.end(); it++){
			int k = it->first;
			float_type alpha_ik = it->second;
                        if( find(yi->begin(), yi->end(), k) == yi->end() ){
                                b[i] = 1.0 - A*alpha_ik;
                                index_b[i] = i;
				act_index_b[i] = k;
				SparseVec* x_i = *(data_it++);
				#ifdef USING_HASHVEC
				int size_vk0 = size_v[k] - 1;
				pair<int, float_type>* vk = v[k];
				for(SparseVec::iterator it2 = x_i->begin(); it2!=x_i->end(); it2++){
					int index_v = 0, J = it2->first;
					find_index(vk, index_v, J, size_vk0, hashindices);
					b[i] += vk[index_v].second*it2->second;
				}
				#else
				float_type* vk = v[k];
				for(SparseVec::iterator it2=x_i->begin(); it2!=x_i->end(); it2++){
					b[i] += vk[it2->first]*it2->second;
				}
				#endif
				i++;
                        }else{
                                c[j] = A*alpha_ik;
                                index_c[j] = j;
				act_index_c[j] = k;
				SparseVec* x_i = *(data_it++);
				#ifdef USING_HASHVEC
				int size_vk0 = size_v[k] - 1;
				pair<int, float_type>* vk = v[k];
				for(SparseVec::iterator it2 = x_i->begin(); it2!=x_i->end(); it2++){
					int index_v = 0, J = it2->first;
					find_index(vk, index_v, J, size_vk0, hashindices);
					c[j] -= vk[index_v].second*it2->second;
				}
				#else
				float_type* vk = v[k];
				for(SparseVec::iterator it2=x_i->begin(); it2!=x_i->end(); it2++)
					c[j] -= vk[it->first]*it->second;
				#endif
				j++;
                        }
		}
		n = i;
		m = j;
		/*for(int i=0;i<n;i++){
			int k = act_index_b[i];
			int ind = invert_index_b[i];
			SparseVec* x_i = x_i_per_class->at(ind);
			#ifdef USING_HASHVEC
			int size_vk0 = size_v[k] - 1;
			pair<int, float_type>* vk = v[k];
			for(SparseVec::iterator it = x_i->begin(); it!=x_i->end(); it++){
				int index_v = 0, j = it->first;
				find_index(vk, index_v, j, size_vk0, hashindices);
				b[i] += vk[index_v].second*it->second;
			}
			#else
			float_type* vk = v[k];
			for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
				b[i] += vk[it->first]*it->second;
			}
			#endif
		}
		for(int i=0;i<m;i++){
			int k = act_index_c[i];
			int ind = invert_index_c[i];
			SparseVec* x_i = x_i_per_class->at(ind);
			#ifdef USING_HASHVEC
			int size_vk0 = size_v[k] - 1;
			pair<int, float_type>* vk = v[k];
			for(SparseVec::iterator it = x_i->begin(); it!=x_i->end(); it++){
				int index_v = 0, j = it->first;
				find_index(vk, index_v, j, size_vk0, hashindices);
				c[i] -= vk[index_v].second*it->second;
			}
			#else
			float_type* vk = v[k];
			for(SparseVec::iterator it=x_i->begin(); it!=x_i->end() ;it++)
				c[i] -= vk[it->first]*it->second;
			#endif
		}*/

		//sort by non-increasing order
		sort(index_b, index_b+n, ScoreComp(b));
		sort(index_c, index_c+m, ScoreComp(c));	

		//partial sums
		float_type* S_b = new float_type[n];
		float_type* S_c = new float_type[m];
		//l_2 residuals
		float_type r_b = 0.0, r_c = 0.0;
		for (int i = 0; i < n; i++){
			b[index_b[i]] /= A;
			r_b += b[index_b[i]]*b[index_b[i]];
			if (i == 0)
				S_b[i] = b[index_b[i]];
			else
				S_b[i] = S_b[i-1] + b[index_b[i]];
		}
		for (int j = 0; j < m; j++){
                        c[index_c[j]] /= A;
			r_c += c[index_c[j]]*c[index_c[j]];
			if (j == 0)
				S_c[j] = c[index_c[j]];
			else
                        	S_c[j] = S_c[j-1] + c[index_c[j]];
                }
		i = 0; j = 0; 
		while (i < n && S_b[i] - (i+1)*b[index_b[i]] <= 0){
			r_b -= b[index_b[i]]*b[index_b[i]];
			i++;
		} 
		while (j < m && S_c[j] - (j+1)*c[index_c[j]] <= 0) { 
                        r_c -= c[index_c[j]]*c[index_c[j]];
                        j++;
                }
		//update for b_{0..i-1} c_{0..j-1}
		//i,j is the indices of coordinate that we will going to include, but not now!
		float_type t = 0.0;
		float_type ans_t_star = 0; 
		float_type ans = INFI; 
		int ansi = i, ansj = j;
		int lasti = 0, lastj = 0;
		do{
			lasti = i; lastj = j;
			float_type l = t;
			if (i == n){
				if (j == m){
					t = C;
				} else {
					t = S_c[j] - (j+1)*c[index_c[j]];
				}
			} else {
				if (j == m){
					t = S_b[i] - (i+1)*b[index_b[i]];
				} else {
					t = S_b[i] - (i+1)*b[index_b[i]];
					if (S_c[j] - (j+1)*c[index_c[j]] < t){ 
                                		t = S_c[j] - (j+1)*c[index_c[j]]; 
                        		}
				}
			}
			if (t > C){
				t = C;
			}
			float_type t_star = (i*S_c[j-1] + j*S_b[i-1])/(i+j);
			if (t_star < l){
				t_star = l;
			}
			if (t_star > t){
				t_star = t;
			}
			float_type candidate = r_b + r_c + (S_b[i-1] - t_star)*(S_b[i-1] - t_star)/i + (S_c[j-1] - t_star)*(S_c[j-1] - t_star)/j;
			if (candidate < ans){
				ans = candidate;
				ansi = i;
				ansj = j;
				ans_t_star = t_star;
			}
			while (i < n && S_b[i] - (i+1)*b[index_b[i]] <= t){
               	         	r_b -= b[index_b[i]]*b[index_b[i]];
               	        	i++;
               	 	}
               	 	while (j < m && S_c[j] - (j+1)*c[index_c[j]] <= t) {
               	         	r_c -= c[index_c[j]]*c[index_c[j]];
               	        	j++;
               	 	}
		} while (i != lasti || j != lastj);
		i = ansi; j = ansj;
		for(int i = 0; i < n; i++){
			int k = act_index_b[index_b[i]];
			if (i < ansi)
				alpha_i_new[k] = -(b[index_b[i]] + (ans_t_star - S_b[ansi-1])/ansi);
			else
				alpha_i_new[k] = 0.0;
		}
		for(int j = 0; j < m; j++){
                        int k = act_index_c[index_c[j]];
			if (j < ansj)
                        	alpha_i_new[k] = c[index_c[j]] + (ans_t_star - S_c[ansj-1])/ansj;
			else
				alpha_i_new[k] = 0.0;
                }

			
		delete[] b; delete[] c;
		delete[] act_index_b; delete[] act_index_c;
		delete[] S_b; delete[] S_c;
		delete[] index_b; delete[] index_c; 
	}

	private:
	Problem* prob;
	float_type C;
	
	vector<SparseVec*>* data_per_class;
	vector<Labels>* labels;
	int D; 
	int N;
	int K;
	float_type* Q_diag;
	vector<pair<int, float_type>>* act_k_index;
	HashClass* hashfunc;
	int* hashindices;
	#ifdef USING_HASHVEC
	pair<int, float_type>** w;
	vector<int>** nnz_index;
	int* size_w;
	pair<int, float_type>** v;
	pair<int, float_type>** alpha;
	int* size_v;
	int* util_v;
	#else
	float_type** w;
	vector<int>** nnz_index;
	float_type** v;
	#endif
		
	int max_iter;
	float_type* grad;
	float_type* Dk;
};
