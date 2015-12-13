#include "util.h"
#include "multi.h"

class PostSolve{
	
	public:
	#ifdef USING_HASHVEC
	PostSolve(Param* param, HashVec** _w, pair<int, float_type>** _alpha, int*& _size_alpha, pair<int, pair<float_type, float_type>>** _v, int*& _size_v){
	#else
	PostSolve(Param* param, HashVec** _w, float_type** _alpha, pair<float_type, float_type>** _v){
	#endif
		prob = param->prob;
		C = param->C;
		
		vector<SparseVec*>* data = &(prob->data);
		labels = &(prob->labels);
		D = prob->D;
		N = prob->N;
		K = prob->K;
		max_iter = param->post_solve_iter;
		hashfunc = new HashClass(K);
		hashindices = hashfunc->hashindices;
		
		// construct data_per_class
		data_per_class = new SparseVec*[N];
		for(int i=0;i<N;i++)
			data_per_class[i] = new SparseVec[K];

		for(int i=0;i<N;i++){
			SparseVec* xi = data->at(i);
			SparseVec* data_per_class_i = data_per_class[i];
			for(SparseVec::iterator it=xi->begin() ;it!=xi->end(); it++){
				int j = it->first;
				double xij = it->second;
				HashVec* wj = _w[j];
				for(HashVec::iterator it2=wj->begin(); it2!=wj->end(); it2++){
					int k = it2->first;
					#ifdef USING_HASHVEC
					int index_alpha = 0;
					int _size_alphai0 = _size_alpha[i] - 1;
					find_index(_alpha[i], index_alpha, k, _size_alphai0, hashindices);
					if (index_alpha == -1) 
						continue;
					if( fabs(_alpha[i][index_alpha].second) > 1e-12)
						data_per_class_i[k].push_back(make_pair(j, xij));
					#else
					if( fabs(_alpha[i][k]) > 1e-12)
						data_per_class_i[k].push_back(make_pair(j, xij));
					#endif
				}
			}
		}
		//initialize alpha and v
		act_k_index = new vector<int>[N];
		
		#ifdef USING_HASHVEC
		alpha = new pair<int, float_type>*[N];
		util_alpha = new int[N];
		size_alpha = new int[N];
		for(int i=0;i<N;i++){
			util_alpha[i] = 0;
			size_alpha[i] = _size_alpha[i]; 
			alpha[i] = new pair<int, float_type>[size_alpha[i]];
			for(int it=0; it < size_alpha[i]; it++){
				alpha[i][it] = _alpha[i][it];
				if (fabs(alpha[i][it].second) > 1e-12){
					act_k_index[i].push_back(alpha[i][it].first);
					util_alpha[i]++;
				}
			}
		}
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
				if (fabs(_v[j][it].second.first) <= param->lambda)
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
				if (fabs(_v[j][it].second.first) <= param->lambda)
					continue;	
				//v[k][j] = _v[j][k];
				//insert (j, _v[j][it]) to v[k]
				int index_v = 0;
				find_index(v[k], index_v, j, size_v[k]-1, hashindices);
				//cout << size_v[k] << " " << index_v << endl;
				v[k][index_v] = make_pair(j, _v[j][it].second.first);
			}
		}
		#else
		alpha = new float_type*[N];
		for(int i=0;i<N;i++){
			alpha[i] = new float_type[K];
			for(int k=0;k<K;k++){
				alpha[i][k] = _alpha[i][k];
				if( fabs(alpha[i][k]) > 1e-12 )
					act_k_index[i].push_back(k);
			}
		}
		v = new float_type*[K]; //w = prox(v);
		for(int k=0;k<K;k++){
			v[k] = new float_type[D];
			for(int j=0;j<D;j++)
				v[k][j] = 0.0;
		}
		for(int j=0;j<D;j++)
			for(HashVec::iterator it=_w[j]->begin(); it!=_w[j]->end(); it++){
				int k = it->first;
				v[k][j] = _v[j][k].first;
			}
		#endif
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
		for(int i=0;i<N;i++)
			delete[] data_per_class[i];
		delete[] data_per_class;
		delete[] act_k_index;
		
		for(int i=0;i<N;i++)
			delete[] alpha[i];
		delete[] alpha;
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
				int act_size = (int)act_k_index[i].size();
				#ifdef USING_HASHVEC
				pair<int, float_type>* alpha_i = alpha[i];
 				int size_alphai = size_alpha[i], size_alphai0 = size_alphai-1;
				#else
				float_type* alpha_i = alpha[i];
				#endif
				
				if( act_size < 2 )
					continue;
				
				//solve subproblem
				subsolve_time -= omp_get_wtime();
				subSolve(i, act_k_index[i], act_size, alpha_i_new);
				subsolve_time += omp_get_wtime();
				
				//maintain v = X^T\alpha 
				maintain_time -= omp_get_wtime();
				SparseVec* x_i_per_class = data_per_class[i];

				for(vector<int>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
					
					int k= *it;
					#ifdef USING_HASHVEC
					int index_alpha = 0;
					find_index(alpha_i, index_alpha, k, size_alphai0, hashindices);
					double alpha_diff = alpha_i_new[k]-alpha_i[index_alpha].second;
					#else
					double alpha_diff = alpha_i_new[k]-alpha_i[k];
					#endif
					SparseVec* x_i = &(x_i_per_class[k]);
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
				for(int r=0;r<act_size;r++){
					int k = act_k_index[i][r];
					#ifdef USING_HASHVEC
					int index_alpha = 0;
					find_index(alpha_i, index_alpha, k, size_alphai0, hashindices);
					alpha_i[index_alpha].second = alpha_i_new[k];
					if (alpha_i[index_alpha].first == -1){
						alpha_i[index_alpha].first = k;
						if ((++util_alpha[i]) > size_alpha[i]*UPPER_UTIL_RATE){
							resize(alpha_i, alpha[i], size_alphai, size_alpha[i], size_alphai0, util_alpha[i], hashindices);
						}
					}
					#else
					alpha_i[k] = alpha_i_new[k];
					#endif
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
		HashVec** w = new HashVec*[D];
		for(int j=0;j<D;j++)
			w[j] = new HashVec();
		for(int k=0;k<K;k++){
			#ifdef USING_HASHVEC
			for(int it = 0; it < size_v[k]; it++){
				int j = v[k][it].first;
				if (j != -1){
					w[j]->insert(make_pair(k, v[k][it].second));	
				}
			}
			#else
			for(int j=0;j<D;j++){	
				float_type wjk = v[k][j];
				if( fabs(wjk) > 1e-12 )
					w[j]->insert(make_pair(k, wjk));
			}
			#endif
		}
		
		float_type d_obj = 0.0;
		int nSV = 0;
		int nnz_w = 0;
		double w_1norm=0.0;
		for(int j=0;j<D;j++){
			for(HashVec::iterator it=w[j]->begin(); it!=w[j]->end(); it++){
				d_obj += it->second*it->second;
				w_1norm += fabs(it->second);
			}
			nnz_w += w[j]->size();
		}
		d_obj/=2.0;
		for(int i=0;i<N;i++){
			Labels* yi = &(labels->at(i));
			#ifdef USING_HASHVEC
			for(int it=0;it<size_alpha[i];it++){
				int k = alpha[i][it].first;
				if (k == -1) 
					continue;
				if(find(yi->begin(), yi->end(), k) ==yi->end())
					d_obj += alpha[i][it].second;
				if( fabs( alpha[i][it].second ) > 1e-12 )
					nSV++;
			}
			#else	
			for(int k=0;k<K;k++){
				if(find(yi->begin(), yi->end(), k) ==yi->end())
					d_obj += alpha[i][k];
				if( fabs( alpha[i][k] ) > 1e-12 )
					nSV++;
			}
			#endif
		}
		cerr << "dual_obj=" << d_obj << endl;
		cerr << "nSV=" << nSV << " (NK=" << N*K << ")"<< endl;
		cerr << "nnz_w=" << nnz_w << " (DK=" << D*K << ")" << endl;
		cerr << "w_1norm=" << w_1norm << endl;
		cerr << "train time=" << endtime-starttime << endl;
		cerr << "subsolve time=" << subsolve_time << endl;
		cerr << "maintain time=" << maintain_time << endl;
		/////////////////////////
		
		//delete algorithm-specific variables
		delete[] alpha_i_new;
		delete[] index;

		return new Model(prob, w);
	}
	
	void subSolve(int I, vector<int>& act_k_index, int act_k_size, float_type* alpha_i_new){
		
		Labels* yi = &(labels->at(I));
		int m = yi->size();
	        int n = act_k_size - m;
		float_type* b = new float_type[n];
		float_type* c = new float_type[m];
		int* act_index_b = new int[n];
		int* act_index_c = new int[m];
		SparseVec* x_i_per_class = data_per_class[I];
		float_type A = Q_diag[I];
		#ifdef USING_HASHVEC
		pair<int, float_type>* alpha_i = alpha[I];
		int size_alphai = size_alpha[I], size_alphai0 = size_alphai-1;
		#else
		float_type* alpha_i = alpha[I];
		#endif
		int i = 0, j = 0;
		int* index_b = new int[n];
		int* index_c = new int[m];
		for(int k=0;k<m+n;k++){
			int p = act_k_index[k];
			#ifdef USING_HASHVEC
			int index_alpha = 0;
			find_index(alpha_i, index_alpha, p, size_alphai0, hashindices);
			if( find(yi->begin(), yi->end(), p) == yi->end() ){
				b[i] = 1.0 - A*alpha_i[index_alpha].second;
				index_b[i] = i;
				act_index_b[i++] = p;
			}else{
				c[j] = A*alpha_i[index_alpha].second;
				index_c[j] = j;
				act_index_c[j++] = p;
			}
			#else
			if( find(yi->begin(), yi->end(), p) == yi->end() ){
				b[i] = 1.0 - A*alpha_i[p];
				index_b[i] = i;
				act_index_b[i++] = p;
			}else{
				c[j] = A*alpha_i[p];
				index_c[j] = j;
				act_index_c[j++] = p;
			}
			#endif
		}

		for(int i=0;i<n;i++){
			int k = act_index_b[i];
			SparseVec* x_i = &(x_i_per_class[k]);
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
			for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++)
				b[i] += vk[it->first]*it->second;
			#endif
		}
		for(int i=0;i<m;i++){
			int k = act_index_c[i];
			SparseVec* x_i = &(x_i_per_class[k]);
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
		}
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
			// l = t; t = min(f_b(i), f_c(j));
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
		//delete[] Dk;	
	}

	private:
	Problem* prob;
	float_type C;
	
	SparseVec** data_per_class ;
	vector<Labels>* labels ;
	int D; 
	int N;
	int K;
	float_type* Q_diag;
	vector<int>* act_k_index;
	#ifdef USING_HASHVEC
	pair<int, float_type>** v;
	pair<int, float_type>** alpha;
	HashClass* hashfunc;
	int* hashindices;
	int* size_v;
	int* size_alpha;
	int* util_v;
	int* util_alpha;
	#else
	float_type** alpha;
	float_type** v;
	#endif
		
	int max_iter;
	float_type* grad;
	float_type* Dk;
};
