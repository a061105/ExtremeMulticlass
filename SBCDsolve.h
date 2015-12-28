#include "util.h"
#include "multi.h"
#include <cassert>
class SBCDsolve{
	
	public:
	SBCDsolve(Param* param){
		
		prob = param->train;
		lambda = param->lambda;
		C = param->C;
		
		data = &(prob->data);
		labels = &(prob->labels);
		D = prob->D;
		N = prob->N;
		K = prob->K;
		max_iter = param->max_iter;
		#ifdef USING_HASHVEC
		//hash_top = 0; hash_bottom = 0;
		hashfunc = new HashClass(K);
		hashindices = hashfunc->hashindices;
		#endif
	}
	
	~SBCDsolve(){
		for(int i=0;i<N;i++)
			delete[] alpha[i];
		delete[] alpha;
		for(int j=0;j<D;j++)
			delete[] v[j];
		delete[] v;
	}
	
/*	#ifdef USING_HASHVEC
	long long hash_top = 0, hash_bottom = 0;
	inline pair<int, float_type>* alpha_i_k(pair<int, float_type>* alpha_i, int i, int act_indexj){
		int size_alphai = size_alpha[i];
		int index_alpha = hashindices[act_indexj] & (size_alphai - 1);
                while (alpha_i[index_alpha].first != -1 && alpha_i[index_alpha].first != act_indexj){
                        index_alpha++;
                        if (index_alpha == size_alphai)
                                index_alpha = 0;
                }
		//if (alpha_i[index_alpha].first == act_indexj)	hash_top++;
		return &(alpha_i[index_alpha]);
	}

	inline pair<int, float_type>* v_j_k(pair<int, float_type>* vj, int j, int act_indexj){
		int size_vj = size_v[j];
		int index_v = hashindices[act_indexj] & (size_vj - 1);
                while (vj[index_v].first != -1 && vj[index_v].first != act_indexj){
                        index_v++;
                        if (index_v == size_vj)
                                index_v = 0;
                }
		//if (vj[index_v].first == act_indexj)	hash_top++;
		return &(vj[index_v]);
	}
	inline void find_index(pair<int, float_type>* l, int& st, const int& index, const int& size0){
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
	inline void resize(pair<int, float_type>*& l, pair<int, float_type>*& L, int& size, int& new_size, int& new_size0, const int& util){
		while (util > UPPER_UTIL_RATE * new_size){
			new_size *= 2;
		}
		pair<int, float_type>* new_l = new pair<int, float_type>[new_size];
		new_size0 = new_size - 1;
		int index_l = 0;
		for (int tt = 0; tt < new_size; tt++){
			new_l[tt] = make_pair(-1, 0.0);
		}
		for (int tt = 0; tt < size; tt++){
			//insert old elements into new array
			pair<int, float_type> p = l[tt];
			if (p.first != -1){
				find_index(new_l, index_l, p.first, new_size0);
				new_l[index_l] = p;
			}
		}
		delete[] l;
		l = new_l;
		size = new_size;
		L = new_l;
	}
	#endif
*/	
	Model* solve(){
		
		//initialize alpha and v ( s.t. v = X^Talpha )
		#ifdef USING_HASHVEC
		alpha = new pair<int, float_type>*[N];
		size_alpha = new int[N];
		util_alpha = new int[N];
		memset(util_alpha, 0, N*sizeof(int));
		for (int i = 0; i < N; i++){
			size_alpha[i] = 1;
			while(size_alpha[i] < K){
				size_alpha[i] = size_alpha[i] << 1;
			}
			alpha[i] = new pair<int, float_type>[size_alpha[i]];
			//memset(alpha[i], 255, sizeof(pair<int, float_type>)*size_alpha[i]);
			for (int k = 0; k < size_alpha[i]; k++){
				alpha[i][k] = make_pair(-1, 0.0);
			}
		}
		v = new pair<int, float_type>*[D];
		size_v = new int[D];
		util_v = new int[D];
		memset(util_v, 0, D*sizeof(int));
		for (int j = 0; j < D; j++){
			size_v[j] = INIT_SIZE;
			v[j] = new pair<int, float_type>[size_v[j]];
			//memset(v[j], 255, sizeof(pair<int, float_type>)*size_v[j]);
			for(int k = 0; k < size_v[j]; k++){
				v[j][k] = make_pair(-1, 0.0);
			}
		}
		#else
		alpha = new float_type*[N];
		for(int i=0;i<N;i++){
			alpha[i] = new float_type[K];
			for(int k=0;k<K;k++)
				alpha[i][k] = 0.0;
		}
		v = new pair<float_type, float_type>*[D]; //w = prox(v);
		for(int j=0;j<D;j++){
			v[j] = new pair<float_type, float_type>[K];
			for(int k=0;k<K;k++)
				v[j][k] = make_pair(0.0, 0.0);
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
		
		//indexes for permutation of [N]
		int* index = new int[N];
		for(int i=0;i<N;i++)
			index[i] = i;
		//initialize active set out of [K] for each sample i
		int* act_k_size = new int[N];
		for(int i=0;i<N;i++)
			act_k_size[i] = K;

		int** act_k_index = new int*[N];
		//int** act_k_hashindex = new int*[N];
		for(int i=0;i<N;i++){
			act_k_index[i] = new int[K];
		//	act_k_hashindex[i] = new int[K];
			for(int k=0;k<K;k++){
				act_k_index[i][k] = k;
		//		act_k_hashindex[i][k] = k & (size_alpha[i]-1);
			}
		}
		//main loop
		double starttime = omp_get_wtime();
		double subsolve_time = 0.0, maintain_time = 0.0;
		float_type* alpha_i_new = new float_type[K];
		int iter = 0;
		while( iter < max_iter ){
			random_shuffle( index, index+N );
			for(int r=0;r<N;r++){
				
				int i = index[r];
				int act_size = act_k_size[i];
				int* act_index = act_k_index[i];
				//int* act_hashindex = new int[K];
				#ifdef USING_HASHVEC
				pair<int, float_type>* alpha_i = alpha[i];
				int size_alphai = size_alpha[i];
				int size_alphai0 = size_alphai - 1;
				int index_alpha = 0, index_v = 0;
				#else
				float_type* alpha_i = alpha[i];
				#endif
				if( act_size < 2 )
					continue;
				
				//solve subproblem
				subsolve_time -= omp_get_wtime();
				subSolve(i, act_index, act_size, alpha_i_new);
				subsolve_time += omp_get_wtime();
				
				//maintain w = prox_{\lambda}( X^T\alpha )
				maintain_time -= omp_get_wtime();
				SparseVec* x_i = data->at(i);
				#ifdef USING_HASHVEC
				float_type* alpha_i_k = new float_type[act_size];
				for(int j = 0; j < act_size; j++){
					int act_indexj = act_index[j];
					/*int index_alpha = hashindices[act_indexj] & size_alphai0;
                                        //hash_top++; hash_bottom++;
                                        if (alpha_i[index_alpha].first != act_indexj)
                                        while (alpha_i[index_alpha].first != act_indexj && alpha_i[index_alpha].first != -1){
						index_alpha++;
                                                index_alpha &= size_alphai0;
						//hash_top++;
                                        }*/
					find_index(alpha_i, index_alpha, act_indexj, size_alphai0, hashindices);
					alpha_i_k[j] = alpha_i_new[act_indexj] - alpha_i[index_alpha].second; 
					//act_hashindex[j] = hashindices[act_indexj] & size_alphai0;
				}
				#endif
				for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){

					int J = it->first;
					float_type f_val = it->second;
					#ifdef USING_HASHVEC
					pair<int, float_type>* vj = v[J];
					int size_vj = size_v[J];
					//float_type upper = size_vj*UPPER_UTIL_RATE;
					int util_vj = util_v[J];
					int size_vj0 = size_vj - 1;
					for(int j = 0; j < act_size; j++){
						int act_indexj = act_index[j];
						//v_j_k(it->first, act_indexj)->second += f_val*(alpha_i_new[act_indexj] - alpha_i_k(i, act_indexj)->second);
						//v_j_k(it->first, act_indexj)->first = act_indexj;
						//int index_v = hashindices[act_indexj] & size_vj0;
						//hash_top++; hash_bottom++;
						//cout << "outside finding " << util_vj << " " << endl;
						find_index(vj, index_v, act_indexj, size_vj0, hashindices);
						vj[index_v].second += f_val*alpha_i_k[j];
						if (vj[index_v].first == -1){
							vj[index_v].first = act_indexj;
							if ((++util_vj) > size_vj * UPPER_UTIL_RATE){
								//resize here
								resize(vj, v[J], size_v[J], size_vj, size_vj0, util_vj, hashindices);
							}
						}
					}
					util_v[J] = util_vj;
					#else
					pair<float_type, float_type>* vj = v[J];
					for(int j=0;j<act_size;j++){
						int k = act_index[j];
						vj[k].first += f_val*(alpha_i_new[k]-alpha_i[k]);
						vj[k].second = prox_l1(vj[k].first, lambda);
					}
					#endif
				}
				//cout << "middle " << endl;
				//update alpha
				#ifdef USING_HASHVEC
				for(int j=0;j<act_size;j++){
					int act_indexj = act_index[j];
					find_index(alpha_i, index_alpha, act_indexj, size_alphai0, hashindices);
					//int index_alpha = hashindices[act_indexj] & size_alphai0;
                                      	//hash_top++; hash_bottom++;
                                       	//while (alpha_i[index_alpha].first != act_indexj && alpha_i[index_alpha].first != -1 ){
                                        //       index_alpha++;
                                        //       if (index_alpha == size_alphai)
                                        //               index_alpha = 0;
                                        //      hash_top++;
                                        //};
					
					if (alpha_i[index_alpha].first == -1 && (alpha_i_new[act_indexj] != 0.0)){
						alpha_i[index_alpha].first = act_indexj;
					};
					alpha_i[index_alpha].second = alpha_i_new[act_indexj];
				}
				if (act_size > size_alphai * UPPER_UTIL_RATE){
					//resize here
					//cout << "enter_alpha" << endl;
					//cout << size_alphai << " " << act_size << endl;
					resize(alpha_i, alpha[i], size_alpha[i], size_alphai, size_alphai0, act_size, hashindices);
					/*while (act_size > size_alphai * UPPER_UTIL_RATE){
						size_alphai = size_alphai << 1;
					}
					//using size_vj as new size_v[j], and size_v[j] is old size
					size_alphai0 = size_alphai - 1;
					pair<int, float_type>* alphai_new = new pair<int, float_type>[size_alphai];
					for (int tt = 0; tt < size_alphai; tt++){
						alphai_new[tt] = make_pair(-1, 0.0);
					}
					for (int tt = 0; tt < size_alpha[i]; tt++){
						//insert vj[tt]
						pair<int, float_type>* temp_pair = &(alpha_i[tt]);
						int k = temp_pair->first;
						if (k != -1){
							int index_alpha = hashindices[k] & size_alphai0;
							if (alphai_new[index_alpha].first != k){
                						while (alphai_new[index_alpha].first != k && alphai_new[index_alpha].first != -1){
                						        index_alpha++;
									index_alpha &= size_alphai0;
                						}
								if (alphai_new[index_alpha].first == -1){
									alphai_new[index_alpha].first = k;
								}
							}
							alphai_new[index_alpha].second = temp_pair->second;
						}
					}
					
					delete[] alpha_i;
					alpha_i = alphai_new; alpha[i] = alphai_new;
					size_alpha[i] = size_alphai;
					//cout << "exit_alpha" << endl;*/
				}
				delete[] alpha_i_k;
				#else
				for(int j=0;j<act_size;j++){
					int k = act_index[j];
					alpha_i[k] = alpha_i_new[k];
				}
				#endif
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
		for (int j = 0; j < D; j++){
			pair<int, float_type>* wj = w[j];
			int size_wj = size_w[j];
			int size_wj0 = size_wj - 1;
			for (int k = 0; k < K; k++){
				int index_v = 0;
				find_index(v[j], index_v, k, size_v[j] - 1, hashindices);
				float_type wjk = v[j][index_v].second;
				if ( fabs(wjk) > 1e-12 ){
					int index_w = 0;
					find_index(wj, index_w, k, size_wj - 1, hashindices);
					wj[index_w].second = wjk;
					if (wj[index_w].first == -1){
						wj[index_w].first = k;
						nnz_index[j]->push_back(k);
						if (size_wj* UPPER_UTIL_RATE < nnz_index[j]->size()){
							int util = nnz_index[j]->size();
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
		for (int j = 0; j < D; j++){
			for (int k = 0; k < K; k++){
				float_type wjk = v[j][k].second;
				if (fabs(wjk) > 1e-12){
					w[j][k] = wjk;
					nnz_index[j]->push_back(k);
				}
			}
		}
		
		#endif
	//	pair<int, float_type>** w = new pair<int, float_type>*[D];
	//	for(int j=0;j<D;j++)
	//		w[j] = new pair<int, float_type>;
	//	for(int j=0;j<D;j++)
	//		for(int k=0;k<K;k++){
	//			#ifdef USING_HASHVEC
	//			int index_v = 0;
	//			find_index(v[j], index_v, k, size_v[j] - 1, hashindices);
	//			float_type wjk = v[j][index_v].second;
	//			//float_type wjk = prox_l1(v_j_k(v[j], j, k)->second, lambda);
	//			#else
	//			float_type wjk = v[j][k].second;
	//			#endif
	//			if( wjk != 0.0 )
	//				w[j]->insert(make_pair(k, wjk));
	//		}
		
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
			
			for(int k=0;k<K;k++){
				
				#ifdef USING_HASHVEC
				int index_alpha = 0;
				find_index(alpha[i], index_alpha, k, size_alpha[i] - 1, hashindices);
				if(find(yi->begin(), yi->end(), k) ==yi->end())
					d_obj += alpha[i][index_alpha].second;
				if( fabs( alpha[i][index_alpha].second ) > 1e-12 )
					nSV++;
				#else
				if(find(yi->begin(), yi->end(), k) ==yi->end())
					d_obj += alpha[i][k];
				if( fabs( alpha[i][k] ) > 1e-12 )
					nSV++;
				#endif
			}
		}
		cerr << "dual_obj=" << d_obj << endl;
		cerr << "nSV=" << nSV << " (NK=" << N*K << ")"<< endl;
		cerr << "nnz_w=" << nnz_w << " (DK=" << D*K << ")" << endl;
		cerr << "w_1norm=" << w_1norm << endl;
		cerr << "train time=" << endtime-starttime << endl;
		cerr << "subsolve time=" << subsolve_time << endl;
		cerr << "maintain time=" << maintain_time << endl;
		//debug
		/////////////////////////
		
		//delete algorithm-specific variables
		delete[] alpha_i_new;
		delete[] act_k_size;
		for(int i=0;i<N;i++)
			delete[] act_k_index[i];
		delete[] act_k_index;
		
		delete[] Q_diag;
		
		#ifdef USING_HASHVEC
		return new Model(prob, nnz_index, w, size_w, hashindices);
		#else
		return new Model(prob, nnz_index, w);
		#endif
	}
	
	void subSolve(int I, int* act_k_index, int act_k_size, float_type* alpha_i_new){
		Labels* yi = &(labels->at(I));
		int m = yi->size(), n = act_k_size - m;
		float_type* b = new float_type[n];
		float_type* c = new float_type[m];
		int* act_index_b = new int[n];
		int* act_index_c = new int[m];
		
		SparseVec* x_i = data->at(I);
		float_type A = Q_diag[I];
		#ifdef USING_HASHVEC
		pair<int, float_type>* alpha_i = alpha[I];
		int size_alphai = size_alpha[I];
		int size_alphai0 = size_alphai - 1;
		int index_alpha = 0;
		#else
		float_type* alpha_i = alpha[I];
		#endif
		int i = 0, j = 0;
		int* index_b = new int[n];
		int* index_c = new int[m];
		for(int k=0;k<m+n;k++){
			int p = act_k_index[k];
			#ifdef USING_HASHVEC
			find_index(alpha_i, index_alpha, p, size_alphai0, hashindices);
			/*int index_alpha = hashindices[p] & size_alphai0;
                        if (alpha_i[index_alpha].first != p){
				//hash_top++; hash_bottom++;
                                while (alpha_i[index_alpha].first != p && alpha_i[index_alpha].first != -1){
                                        index_alpha++; index_alpha &= size_alphai0;
				//	hash_top++;
                                }
                        }*/
			
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

		for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
			int fea_ind = it->first;
			float_type fea_val = it->second;
			#ifdef USING_HASHVEC
			pair<int, float_type>* vj = v[fea_ind];
			int size_vj = size_v[fea_ind];
			int size_vj0 = size_vj - 1;
			int index_v = 0;
			#else
			pair<float_type, float_type>* vj = v[fea_ind];
			#endif
			for(int i = 0; i < n; i++){
				int k = act_index_b[i];
				#ifdef USING_HASHVEC
				/*int index_v = hashindices[k] & size_vj0;
				//int first = vj[index_v].first;
				if (vj[index_v].first != k){
					//hash_top++; hash_bottom++;
					if (vj[index_v].first != -1){
						do{
                				        index_v++; index_v &= size_vj0;
							//hash_top++;
                				}while (vj[index_v].first != k && vj[index_v].first != -1);
					}
				}*/
				find_index(vj, index_v, k, size_vj0, hashindices);
				float_type vjk = vj[index_v].second;
				if (fabs(vjk) > lambda){
					if (vjk > 0)
						b[i] += (vjk-lambda)*fea_val;
					else
						b[i] += (vjk+lambda)*fea_val;
				}
				#else
				float_type wjk = vj[k].second;
				b[i] += wjk*fea_val;
				#endif
			}
			for(int j = 0; j < m; j++){
				int k = act_index_c[j];
				#ifdef USING_HASHVEC
				/*int index_v = hashindices[k] & size_vj0;
				if (vj[index_v].first != k){
					//hash_top++; hash_bottom++;
					if (vj[index_v].first != -1){
						do{
                				        index_v++; index_v &= size_vj0;
							//hash_top++;
                				}while (vj[index_v].first != k && vj[index_v].first != -1);
					}
				}*/
				find_index(vj, index_v, k, size_vj0, hashindices);
				float_type vjk = vj[index_v].second;	
				if( fabs(vjk) > lambda ){
					if( vjk > 0 )
						c[j] -= (vjk-lambda)*fea_val;
					else
						c[j] -= (vjk+lambda)*fea_val;
				}
				#else
				c[j] -= vj[k].second*fea_val;
				#endif
			}
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
	float_type lambda;
	float_type C;
	vector<SparseVec*>* data ;
	vector<Labels>* labels ;
	int D; 
	int N;
	int K;
	float_type* Q_diag;
	#ifdef USING_HASHVEC
	pair<int, float_type>** w;
	vector<int>** nnz_index;
	int* size_w;
	pair<int, float_type>** v;
	pair<int, float_type>** alpha;
	HashClass* hashfunc;
	int* hashindices;
	int* size_v;
	int* size_alpha;
	int* util_v;
	int* util_alpha;
	#else
	float_type** w;
	vector<int>** nnz_index;
	float_type** alpha;
	pair<float_type, float_type>** v;
	#endif
		
	int max_iter;
	float_type* grad;
	float_type* Dk;
};
