#include "util.h"
#include "multi.h"
#include "newHash.h"
#include <cassert>
#define loc(k) k*split_up_rate/K

class SplitOracleActBCD{
	
	public:
	SplitOracleActBCD(Param* param){
		prob = param->prob;
		lambda = param->lambda;
		C = param->C;
		N = prob->N;
		D = prob->D;
		K = prob->K;
		hashfunc = new HashClass(K);
		hashindices = hashfunc->hashindices;

		//sampling 	
		speed_up_rate = param->speed_up_rate;	
		split_up_rate = param->split_up_rate;
		using_importance_sampling = param->using_importance_sampling;	
		max_select = param->max_select;
		prod = new float_type[K];
		
		data = &(prob->data);
		//compute l_1 norm of every feature x_i
		cdf_sum = new vector<float_type>();
		for(int i = 0; i < N; i++){
			SparseVec* xi = data->at(i);
			float_type _cdf = 0.0;
			for (SparseVec::iterator it = xi->begin(); it < xi->end(); it++){
				_cdf += fabs(it->second);
			}
			cdf_sum->push_back(_cdf);
		}
		labels = &(prob->labels);
		max_iter = param->max_iter;
		
		//compute location of k
		/*loc = new int[K];
		for (int k = 0; k < K; k++){
			loc[k] = loc(k);
		}*/
		//DEBUG
	}
	
	~SplitOracleActBCD(){
		for(int i=0;i<N;i++)
			delete[] alpha[i];
		delete[] alpha;
		for(int j=0;j<D;j++)
			delete[] v[j];
		delete[] v;
		#ifdef USING_HASHVEC
		delete[] size_v;
		delete[] size_alpha;
		#endif
	}

	
	

	Model* solve(){
		//initialize alpha and v ( s.t. v = X^Talpha )
		#ifdef USING_HASHVEC
                alpha = new pair<int, float_type>*[N];
                size_alpha = new int[N];
                util_alpha = new int[N];
                memset(util_alpha, 0, N*sizeof(int));
                for (int i = 0; i < N; i++){
                        size_alpha[i] = INIT_SIZE;
                        alpha[i] = new pair<int, float_type>[size_alpha[i]];
                        for (int k = 0; k < size_alpha[i]; k++){
                                alpha[i][k] = make_pair(-1, 0.0);
                        }
                }
                v = new pair<int, pair<float_type, float_type>>*[D];
                size_v = new int[D];
                util_v = new int[D];
                memset(util_v, 0, D*sizeof(int));
                for (int j = 0; j < D; j++){
                        size_v[j] = INIT_SIZE;
                        v[j] = new pair<int, pair<float_type,float_type>>[size_v[j]];
                        for(int k = 0; k < size_v[j]; k++){
                                v[j][k] = make_pair(-1, make_pair(0.0, 0.0));
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
			for(int k=0;k<K;k++){
				v[j][k] = make_pair(0.0, 0.0);
			}
		}
		#endif
		//initialize non-zero index array w
		w_hash_nnz_index = new pair<int, float_type>**[D];
		size_w = new int*[D]; util_w = new int*[D];
		for(int j=0;j<D;j++){
			w_hash_nnz_index[j] = new pair<int, float_type>*[split_up_rate];
			size_w[j] = new int[split_up_rate];
			util_w[j] = new int[split_up_rate];
			for(int S=0;S < split_up_rate; S++){
				size_w[j][S] = INIT_SIZE;
				util_w[j][S] = 0;
				w_hash_nnz_index[j][S] = new pair<int, float_type>[size_w[j][S]];
				for(int k = 0; k < size_w[j][S]; k++){
					w_hash_nnz_index[j][S][k] = make_pair(-1, 0.0);
				}
			}
		}
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
		vector<int>* act_k_index = new vector<int>[N];
		for(int i=0;i<N;i++){
			Labels* yi = &(labels->at(i));
			for (Labels::iterator it = yi->begin(); it < yi->end(); it++){
				act_k_index[i].push_back(*it);
				#ifdef USING_HASHVEC
				util_alpha[i]++;
				#endif
			}
		}
		
		//main loop
		double starttime = omp_get_wtime();
		double search_time=0.0, subsolve_time=0.0, maintain_time=0.0;
		double last_search_time = 0.0, last_subsolve_time = 0.0, last_maintain_time = 0.0;
		float_type* alpha_i_new = new float_type[K];
		int iter = 0;
		while( iter < max_iter ){
			
			random_shuffle( index, index+N );
			for(int r=0;r<N;r++){	
				
				int i = index[r];
				auto alpha_i = alpha[i];
				#ifdef USING_HASHVEC
				int size_alphai = size_alpha[i];
				int size_alphai0 = size_alphai - 1;
				int index_alpha = 0, index_v = 0;
				#endif
				//search active variable
				search_time -= omp_get_wtime();
				if (using_importance_sampling)
					search_active_i_importance( i, act_k_index[i]);
				else
					search_active_i_uniform(i, act_k_index[i]);
				
				search_time += omp_get_wtime();
				//solve subproblem
				if( act_k_index[i].size() < 2 )
					continue;
					
				subsolve_time -= omp_get_wtime();
				subSolve(i, act_k_index[i], alpha_i_new);
				subsolve_time += omp_get_wtime();
				
				//maintain v =  X^T\alpha;  w = prox_{l1}(v);
				SparseVec* x_i = data->at(i);
				Labels* yi = &(labels->at(i));
				maintain_time -= omp_get_wtime();
				#ifdef USING_HASHVEC
				float_type* alpha_i_k = new float_type[act_k_index[i].size()];
				for(int j = 0; j < act_k_index[i].size(); j++){
                                        int act_indexj = act_k_index[i][j];
                                        find_index(alpha_i, index_alpha, act_indexj, size_alphai0, hashindices);
                                        alpha_i_k[j] = alpha_i_new[act_indexj] - alpha_i[index_alpha].second;
                                }	
				#endif
				for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
					int J = it->first; 
					float_type f_val = it->second;
					//auto vj = v[J];
					#ifdef USING_HASHVEC
					pair<int, pair<float_type, float_type>>* vj = v[J];
					int size_vj = size_v[J];
					int util_vj = util_v[J];
					int size_vj0 = size_vj - 1;
					for (int j = 0; j < act_k_index[i].size(); j++){
						int k = act_k_index[i][j];
						float_type delta_alpha = alpha_i_k[j];
						if( fabs(delta_alpha) < 1e-12 )
							continue;
						//update v
						
						find_index(vj, index_v, k, size_vj0, hashindices);
						float_type vjk = vj[index_v].second.first + f_val*delta_alpha;
						float_type wjk_old = vj[index_v].second.second;
						float_type wjk = prox_l1(vjk, lambda);
						vj[index_v].second = make_pair(vjk, wjk);
						if (vj[index_v].first == -1){
							vj[index_v].first = k;
							if ((++util_v[J]) > size_vj * UPPER_UTIL_RATE){
								resize(vj, v[J], size_v[J], size_vj, size_vj0, util_v[J], hashindices);
							}
						}
						if ( wjk_old != wjk ){
							int index_w = 0, lock = loc(k);
                                                       	pair<int, float_type>* wj = w_hash_nnz_index[J][lock];
                                                        find_index(wj, index_w, k, size_w[J][lock]-1, hashindices);
							wj[index_w].second = wjk;
							if (wj[index_w].first == -1){
								wj[index_w].first = k;
								if ((++util_w[J][lock]) > size_w[J][lock] * UPPER_UTIL_RATE){
									int size_wjl = size_w[J][lock];
									int size_wjl0 = size_wjl - 1;
									resize(wj, w_hash_nnz_index[J][lock], size_wjl, size_w[J][lock], size_wjl0, util_w[J][lock], hashindices);
								}
							}
                                                }
					}	
					#else
					pair<float_type, float_type>* vj = v[J];
					pair<int, float_type>** wJ = w_hash_nnz_index[J];
					for(vector<int>::iterator it2 = act_k_index[i].begin(); it2 < act_k_index[i].end(); it2++){
						int k = *it2;
						float_type delta_alpha = (alpha_i_new[k]-alpha_i[k]);
						if( fabs(delta_alpha) < 1e-12 )
							continue;
						//update v
						pair<float_type, float_type> vjk_wjk = vj[k];
						//float_type vjk = vjk_wjk->first + f_val*delta_alpha;
						//float_type wjk = prox_l1(vjk, lambda);
						float_type wjk_old = vjk_wjk.second;
						vjk_wjk.first += f_val*delta_alpha;
						vjk_wjk.second = prox_l1(vjk_wjk.first, lambda);
						vj[k] = vjk_wjk;
						// *(vjk_wjk) = make_pair(vjk, wjk);
						if ( wjk_old != vjk_wjk.second ){
							int index_w = 0, lock = loc(k);
                                                       	pair<int, float_type>* wj = wJ[lock];
                                                        find_index(wj, index_w, k, size_w[J][lock]-1, hashindices);
							wj[index_w].second = vjk_wjk.second;
							if (wj[index_w].first == -1){
								wj[index_w].first = k;
								if ((++util_w[J][lock]) > size_w[J][lock] * UPPER_UTIL_RATE){
									int size_wjl = size_w[J][lock];
									int size_wjl0 = size_wjl - 1;
									resize(wj, w_hash_nnz_index[J][lock], size_wjl, size_w[J][lock], size_wjl0, util_w[J][lock], hashindices);
								}
							}
                                                }
					}
					#endif
				}
				//update alpha
				bool has_zero=0;
				for(vector<int>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
					int k = *it;
					#ifdef USING_HASHVEC
					find_index(alpha_i, index_alpha, k, size_alphai0, hashindices);
					if (fabs(alpha_i_new[k]) > 1e-12){
						alpha_i[index_alpha] = make_pair(k, alpha_i_new[k]);
						if ((++util_alpha[i]) > size_alphai * UPPER_UTIL_RATE){
							//cout << "success alpha" << endl;
							resize(alpha_i, alpha[i], size_alpha[i], size_alphai, size_alphai0, util_alpha[i], hashindices);
						}
					} else {
						has_zero = true;
					}
					#else
					alpha_i[k] = alpha_i_new[k];
					has_zero |= (fabs(alpha_i_new[k])<1e-12);
					#endif
				}
				//shrink act_k_index
				if( has_zero ){
					#ifdef USING_HASHVEC
					//util_alpha[i] = 0;
					#endif
					//cerr << "before size=" << act_k_index[i].size() << endl;
					vector<int> tmp_vec;
					tmp_vec.reserve(act_k_index[i].size());
					for(vector<int>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
						int k = *it;
					//	cerr << alpha_i[k] << " ";
						if( fabs(alpha_i_new[k]) > 1e-12 || find(yi->begin(), yi->end(), k)!=yi->end() ){
							tmp_vec.push_back(k);
							#ifdef USING_HASHVEC
					//		util_alpha[i]++;
							#endif
						}
					}
					//cerr << endl;
					act_k_index[i].clear();
					act_k_index[i] = tmp_vec;
					//cerr << "after size=" << act_k_index[i].size() << endl;
				}
				maintain_time += omp_get_wtime();
			}
		
			//throw out zeros elements
			maintain_time -= omp_get_wtime();
			//vector<int> tmp_vec;
			for(int j=0;j<D;j++){
				//auto vj=v[j];
				for(int S=0; S < split_up_rate; S++){
					//pair<int, float_type>* wj = w_hash_nnz_index[j][S];
					//tmp_vec.clear();
					/*for(int k = 0; k < size_w[j][S]; k++){
						if (wj[k].second == 0.0)
							wj[k].first = -1;
					}
					(*wj) = tmp_vec;*/
					trim(w_hash_nnz_index[j][S], size_w[j][S], util_w[j][S], hashindices);
				}
			}
			maintain_time += omp_get_wtime();

			if( iter % 1 == 0 ){
				cerr << "." ;
				int nnz_a_i = 0;
				for(int i=0;i<N;i++){
					nnz_a_i += act_k_index[i].size();	
				}
				cerr << "nnz_a_i="<< ((float_type)nnz_a_i/N) << "  \t";
				int nnz_w_j = 0;
				for(int j=0;j<D;j++){
					for(int S=0;S < split_up_rate; S++){
						nnz_w_j += util_w[j][S]; //w_nnz_index[j][S]->size();
					}
				}
				cerr << "nnz_w_j=" << ((float_type)nnz_w_j/D) << "  \t";
				cerr << "search=" << search_time-last_search_time << "  \t";
				cerr << "subsolve=" << subsolve_time-last_subsolve_time << "  \t";
				cerr << "maintain=" << maintain_time-last_maintain_time << endl;
				last_search_time = search_time; last_subsolve_time = subsolve_time; last_maintain_time = maintain_time;
			}
				
			iter++;
		}
		double endtime = omp_get_wtime();
		cerr << endl;
		
		float_type d_obj = 0.0;
		int nSV = 0;
		int nnz_w = 0;
		int jk=0;
		for(int j=0;j<D;j++){
			for(int S=0;S<split_up_rate;S++){
				//for (vector<int>::iterator it=w_nnz_index[j][S]->begin(); it!=w_nnz_index[j][S]->end(); it++){
				//	#ifdef USING_HASHVEC
				//	int index_v = 0;
				//	find_index(v[j], index_v, *it, size_v[j]-1);
				//	float_type wjk = prox_l1(v[j][index_v].second, lambda);
				//	#else
				//	float_type wjk = v[j][*it].second; //prox_l1(v[j][*it], lambda);
				//	#endif
				//	d_obj += wjk*wjk;//W[j][*it]*W[j][*it];
				//}
				//nnz_w+=w_nnz_index[j][S]->size();
				pair<int, float_type>* wjS = w_hash_nnz_index[j][S];
				for (int k = 0; k < size_w[j][S]; k++){
					float_type wjk = wjS[k].second;
					//.second is 0 by default
					d_obj += wjk*wjk;
				}
				nnz_w+= util_w[j][S];
			}
		}
		d_obj/=2.0;
		for(int i=0;i<N;i++){
			Labels* yi = &(labels->at(i));
			for(int k=0;k<K;k++){
				#ifdef USING_HASHVEC
				int index_alpha = 0;
				find_index(alpha[i], index_alpha, k, size_alpha[i] - 1, hashindices);
				#endif
				if(find(yi->begin(), yi->end(), k) == yi->end()){
					#ifdef USING_HASHVEC
					d_obj += alpha[i][index_alpha].second;
					#else
					d_obj += alpha[i][k];
					#endif
				}
				#ifdef USING_HASHVEC
				if ( fabs(alpha[i][index_alpha].second) > 1e-12 )
					nSV++;
				#else
				if( fabs(alpha[i][k]) > 1e-12 )
					nSV++;
				#endif
			}
		}
		cerr << "dual_obj=" << d_obj << endl;
		cerr << "nSV=" << nSV << " (NK=" << N*K << ")"<< endl;
		cerr << "nnz_w=" << nnz_w << " (DK=" << D*K << ")" << endl;
		cerr << "train time=" << endtime-starttime << endl;
		cerr << "search time=" << search_time << endl;
		cerr << "subsolve time=" << subsolve_time << endl;
		cerr << "maintain time=" << maintain_time << endl;
		//delete algorithm-specific variables
		delete[] alpha_i_new;
		delete[] act_k_index;
		delete[] Q_diag;
		delete[] prod;
		delete cdf_sum;
		w_temp = new HashVec*[D];
		for(int j = 0; j < D; j++){
			w_temp[j] = new HashVec();
			for(int S=0;S<split_up_rate;S++){
				//for (vector<int>::iterator it=w_nnz_index[j][S]->begin(); it!=w_nnz_index[j][S]->end(); it++){
				//	int k = *it;
				//	#ifdef USING_HASHVEC
				//	int index_v = 0;
				//	find_index(v[j], index_v, k, size_v[j]-1);
				//	w_temp[j]->insert(make_pair(k, prox_l1(v[j][index_v].second, lambda)));
				//	#else
				//	w_temp[j]->insert(make_pair(k, v[j][k].second)); //prox_l1(v[j][k], lambda)));
				//	#endif
				//}
				pair<int, float_type>* wjS = w_hash_nnz_index[j][S];
				for (int k = 0; k < size_w[j][S]; k++){
					if (wjS[k].first != -1 && wjS[k].second != 0.0){
						w_temp[j]->insert(wjS[k]);
					}
				}
			}
		}
		for(int j = 0; j < D; j++){
			delete[] size_w[j];
			delete[] util_w[j];
			for (int S = 0; S < split_up_rate; S++){
				delete[] w_hash_nnz_index[j][S];
			}
			delete[] w_hash_nnz_index[j];
		}
		delete[] w_hash_nnz_index;
		delete[] size_w;
		delete[] util_w;
		return new Model(prob, w_temp); //v is w
	}
	void subSolve(int I, vector<int>& act_k_index, float_type* alpha_i_new){
	
			
		Labels* yi = &(labels->at(I));
		int m = yi->size(), n = act_k_index.size() - m;
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
                                //      hash_top++;
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
			//auto vj = v[fea_ind];
			#ifdef USING_HASHVEC
                        pair<int, pair<float_type, float_type>>* vj = v[fea_ind];
                        int size_vj = size_v[fea_ind];
                        int size_vj0 = size_vj - 1;
                        int index_v = 0;
                        #else
                        pair<float_type, float_type>* vj = v[fea_ind];
                        #endif
			for(int i = 0; i < n; i++){
				//int k = act_index_b[i];
				#ifdef USING_HASHVEC
                                find_index(vj, index_v, act_index_b[i], size_vj0, hashindices);
                                float_type wjk = vj[index_v].second.second;
                                b[i] += wjk*fea_val;
                                #else
                                //float_type vjk = vj[k];
                                b[i] += vj[act_index_b[i]].second*fea_val;
                                #endif
			}
			for(int j = 0; j < m; j++){
				//int k = act_index_c[j];
				#ifdef USING_HASHVEC
                                find_index(vj, index_v, act_index_c[j], size_vj0, hashindices);
                                float_type wjk = vj[index_v].second.second;
                                c[j] -= wjk*fea_val;
                                #else
                                //float_type vjk = vj[k];
                                c[j] -= vj[act_index_c[j]].second*fea_val;
				#endif
			}
		}
			
		sort(index_b, index_b+n, ScoreComp(b));
		sort(index_c, index_c+m, ScoreComp(c));				
		float_type* S_b = new float_type[n];
		float_type* S_c = new float_type[m];
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
			assert(t >= 0 && l >= 0);
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
		
	void search_active_i_importance( int i, vector<int>& act_k_index ){
		int S = rand()%split_up_rate;
                //compute <xi,wk> for k=1...K
                Labels* yi = &(labels->at(i));
		memset(prod, 0, sizeof(float_type)*K);
                SparseVec* xi = data->at(i);
		int nnz = xi->size();
		for(int j = 0; j < act_k_index.size(); j++){
			prod[act_k_index[j]] = -INFI;
		}
		for (Labels::iterator it = yi->begin(); it < yi->end(); it++){
			prod[*it] = -INFI;
		}
		int n = nnz/speed_up_rate;
		float_type th = -n/(1.0*nnz);
		vector<float_type>* rand_nums = new vector<float_type>();
		for (int tt = 0; tt < n; tt++){
			rand_nums->push_back(((float_type)rand()/(RAND_MAX)));
		}
		sort(rand_nums->begin(), rand_nums->end()); 
		#ifdef MULTISELECT
		int* max_indices = new int[max_select+1];
		for(int ind = 0; ind <= max_select; ind++){
			max_indices[ind] = -1;
		}
		#endif
		#ifndef MULTISELECT
		int max_index = 0;
		#endif
		SparseVec::iterator current_index = xi->begin();
		float_type current_sum = current_index->second;
		vector<float_type>::iterator current_rand_index = rand_nums->begin();
		float_type cdf_sumi = cdf_sum->at(i);
			
		while (current_rand_index < rand_nums->end()){
			while (current_sum < (*current_rand_index)*cdf_sumi){
				current_index++;
				current_sum += current_index->second;
			}
			float_type xij = 0.0;
			while (current_rand_index < rand_nums->end() && current_sum >= (*current_rand_index)*cdf_sumi ){
				xij = xij + 1.0;
				current_rand_index++;
			}
                        xij *= cdf_sumi*((current_index->second > 0.0)?1:(-1));
			int j = current_index->first;
			if (util_w[j][S] == 0) continue;
			//vector<int>* wj = w_nnz_index[j][S];
			pair<int, float_type>* wjS = w_hash_nnz_index[j][S];
			int size_wjS = size_w[j][S];
			int k = 0, ind = 0;
			float_type wjk = 0.0;
			//auto vj = v[j];
			//vector<int>::iterator it2 = wj->begin();
			//vector<int>::iterator tail = wj->end();
                        //for(vector<int>::iterator it2 = wj->begin(); it2<wj->end(); it2++ ){
			//pair<int, float_type>* p;
			//cout << util_w[j][S] << "/" << size_wjS << endl;
			for (int it2 = 0; it2 < size_wjS; it2++){
				//p = &(wjS[it2]);
				//wjk = wjS[it2].second;
				//if (wjk == 0.0) continue;
				k = wjS[it2].first;
				if (k == -1)
					continue;
				//cout << wjk << endl;
				//assert(k != -1);
                                prod[k] += wjS[it2].second * xij;
				#ifndef MULTISELECT
				if (prod[max_index] < prod[k]){
					max_index = k;
				}
				#endif
				#ifdef MULTISELECT
				if (prod[k] > th){
					ind = 0;
					while (ind < max_select && max_indices[ind] != -1 && max_indices[ind] != k){
						ind++;
					}
					max_indices[ind] = k;
					//try move to left
					while (ind > 0 && prod[max_indices[ind]] > prod[max_indices[ind-1]]){
						//using k as temporary variables
						k = max_indices[ind];
						max_indices[ind] = max_indices[ind-1];
						max_indices[ind-1] = k;
					}
					//try move to right
					while (ind < max_select-1 && max_indices[ind+1] != -1 && prod[max_indices[ind+1]] > prod[max_indices[ind]]){
                                                //using k as temporary variables
                                                k = max_indices[ind];
                                                max_indices[ind] = max_indices[ind+1];
                                                max_indices[ind+1] = k;
                                        }
				}
				#endif
				//it2++;
			}
			//wj->erase(tail, wj->end());
                }
		rand_nums->clear();
		#ifdef MULTISELECT
		for (int j = 0; j < max_select; j++){
			if (max_indices[j] != -1 && prod[max_indices[j]] > 0.0) 
				continue;
			for (int r = 0; r < K; r++){
				int k = rand() % K;
				if (prod[k] == 0){
					bool flag = false;
					for (int ind = 0; ind < max_select; ind++){
						if (max_indices[ind] == k){
							flag = true;
							break;
						}
					}
					if (!flag){
						max_indices[j] = k;
						break;
					}
				}
			}
		}
		for(int ind = 0; ind < max_select; ind++){
			if (max_indices[ind] != -1 && prod[max_indices[ind]] > th){
				act_k_index.push_back(max_indices[ind]);
			}
		}
		#endif
		#ifndef MULTISELECT
		if (prod[max_index] < 0){
			for (int r = 0; r < K; r++){
				int k = rand() % K;
				if (prod[k] == 0){
					max_index = k;
					break;
				}
			}
		}
		if (prod[max_index] > th){
			for (int k = 0; k < act_k_index.size(); k++){
				//cerr << prod[max_index] << " " << th<< endl;
				assert(act_k_index[k] != max_index);
			}
			act_k_index.push_back(max_index);
//			cerr << "yes" << endl;
		} else{
//			cerr << "no" << endl;
		}
 
		#endif
        }

	void search_active_i_uniform(int i, vector<int>& act_k_index){
		int S = rand()%split_up_rate;
                //compute <xi,wk> for k=1...K
                Labels* yi = &(labels->at(i));
		memset(prod, 0, sizeof(float_type)*K);
                SparseVec* xi = data->at(i);
		int nnz = xi->size();
		for(int j = 0; j < act_k_index.size(); j++){
			prod[act_k_index[j]] = -INFI;
		}
		for (Labels::iterator it = yi->begin(); it < yi->end(); it++){
			prod[*it] = -INFI;
		}
		int n = nnz/speed_up_rate;
		float_type th = -n/(1.0*nnz);
		vector<float_type>* rand_nums = new vector<float_type>();
		for (int tt = 0; tt < n; tt++){
			rand_nums->push_back(((float_type)rand()/(RAND_MAX)));
		}
		sort(rand_nums->begin(), rand_nums->end()); 
		#ifdef MULTISELECT
		int* max_indices = new int[max_select+1];
		for(int ind = 0; ind <= max_select; ind++){
			max_indices[ind] = -1;
		}
		#endif
		#ifndef MULTISELECT
		int max_index = 0;
		#endif
		random_shuffle(xi->begin(), xi->end());
		for (SparseVec::iterator current_index = xi->begin(); current_index < xi->begin() + n; current_index++){
			float_type xij = current_index->second;
			int j = current_index->first;
			if (util_w[j][S] == 0) continue;
			pair<int, float_type>* wjS = w_hash_nnz_index[j][S];
			int size_wjS = size_w[j][S];
			int k = 0, ind = 0;
			float_type wjk = 0.0;
			for (int it2 = 0; it2 < size_wjS; it2++){
				k = wjS[it2].first;
				if (k == -1)
					continue;
				//cout << wjk << endl;
				//assert(k != -1);
                                prod[k] += wjS[it2].second * xij;
				#ifndef MULTISELECT
				if (prod[max_index] < prod[k]){
					max_index = k;
				}
				#endif
				#ifdef MULTISELECT
				if (prod[k] > th){
					ind = 0;
					while (ind < max_select && max_indices[ind] != -1 && max_indices[ind] != k){
						ind++;
					}
					max_indices[ind] = k;
					//try move to left
					while (ind > 0 && prod[max_indices[ind]] > prod[max_indices[ind-1]]){
						//using k as temporary variables
						k = max_indices[ind];
						max_indices[ind] = max_indices[ind-1];
						max_indices[ind-1] = k;
					}
					//try move to right
					while (ind < max_select-1 && max_indices[ind+1] != -1 && prod[max_indices[ind+1]] > prod[max_indices[ind]]){
                                                //using k as temporary variables
                                                k = max_indices[ind];
                                                max_indices[ind] = max_indices[ind+1];
                                                max_indices[ind+1] = k;
                                        }
				}
				#endif
			}
                }
		rand_nums->clear();
		#ifdef MULTISELECT
		for (int j = 0; j < max_select; j++){
			if (max_indices[j] != -1 && prod[max_indices[j]] > 0.0) 
				continue;
			for (int r = 0; r < K; r++){
				int k = rand() % K;
				if (prod[k] == 0){
					bool flag = false;
					for (int ind = 0; ind < max_select; ind++){
						if (max_indices[ind] == k){
							flag = true;
							break;
						}
					}
					if (!flag){
						max_indices[j] = k;
						break;
					}
				}
			}
		}
		for(int ind = 0; ind < max_select; ind++){
			if (max_indices[ind] != -1 && prod[max_indices[ind]] > th){
				act_k_index.push_back(max_indices[ind]);
			}
		}
		#endif
		#ifndef MULTISELECT
		if (prod[max_index] < 0){
			for (int r = 0; r < K; r++){
				int k = rand() % K;
				if (prod[k] == 0){
					max_index = k;
					break;
				}
			}
		}
		if (prod[max_index] > th){
			for (int k = 0; k < act_k_index.size(); k++){
				//cerr << prod[max_index] << " " << th<< endl;
				assert(act_k_index[k] != max_index);
			}
			act_k_index.push_back(max_index);
//			cerr << "yes" << endl;
		} else{
//			cerr << "no" << endl;
		}
 
		#endif
		/*int S = rand()%split_up_rate;
		Labels* yi = &(labels->at(i));
                memset(prod, 0, sizeof(float_type)*K);
                SparseVec* xi = data->at(i);
                int nnz = xi->size();
                for(int j = 0; j < act_k_index.size(); j++){
                        prod[act_k_index[j]] = -INFI;
                }
		for(Labels::iterator it = yi->begin(); it < yi->end(); it++){
                	prod[*it] = -INFI;
		}
                int n = nnz/speed_up_rate;
                vector<float_type>* rand_nums = new vector<float_type>();
                for (int tt = 0; tt < n; tt++){
                        rand_nums->push_back(((float_type)rand()/(RAND_MAX)));
                }
                sort(rand_nums->begin(), rand_nums->end());
                int max_index = 0;
		random_shuffle(xi->begin(), xi->end());
                SparseVec::iterator current_index = xi->begin();
                for (int t = 0; t < n; t++){
                        float_type xij = current_index->second;
                        int j = current_index->first;
                        current_index++;
                        vector<int>* wj = w_nnz_index[j][S];
                        int k = 0;
                        float_type wjk = 0.0;
                        for(vector<int>::iterator it2 = wj->begin(); it2<wj->end(); it2++ ){
                                k = *it2;
                                wjk = prox_l1(v[j][k], lambda);
                                if (wjk == 0.0){
                                        *it2=*(wj->end()-1);
                                        wj->erase(wj->end()-1);
                                        it2--;
                                        continue;
                                }
                                prod[k] += wjk * xij;
                                if (prod[k] > prod[max_index]){
                                        max_index = k;
                                }
			}
		}
		float_type th = -n/(1.0*nnz);
                if (prod[max_index] < 0){
                        for(int k = 0; k < K; k++){
                                int r = rand()%K;
                                if (prod[r] == 0){
                                        max_index = r;
                                        break;
                                }
                        }
                }
		if (prod[max_index] > th){
                        act_k_index.push_back(max_index);
                }*/
	}
	
	private:
	Problem* prob;
	float_type lambda;
	float_type C;
	vector<SparseVec*>* data;
	vector<Labels>* labels;
	int D; 
	int N;
	int K;
	float_type* Q_diag;
        HashClass* hashfunc;
        int* hashindices;
	vector<float_type>* cdf_sum;
	HashVec** w_temp;
	pair<int, float_type>*** w_hash_nnz_index;
	int** size_w;
	int** util_w;
	int max_iter;
	vector<int>* k_index;
		
	//sampling 
	bool using_importance_sampling;
	int max_select;
	int speed_up_rate, split_up_rate;	
	float_type* prod;
	//int* loc;
	public:
	#ifdef USING_HASHVEC
	pair<int, pair<float_type, float_type> >** v;
        pair<int, float_type>** alpha;
        int* size_v;
        int* size_alpha;
        int* util_v;
        int* util_alpha;
	#else
	float_type** alpha;
	pair<float_type, float_type>** v;
	#endif
};
