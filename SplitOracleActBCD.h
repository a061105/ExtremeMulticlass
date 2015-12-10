#include "util.h"
#include "multi.h"
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
		#ifdef USING_HASHVEC
		hashfunc = new HashClass(K);
		hashindices = hashfunc->hashindices;
		#endif

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
		
		//DEBUG
	}
	
	~SplitOracleActBCD(){
	}

	#ifdef USING_HASHVEC
	long long hash_top = 0, hash_bottom = 0;
        inline pair<int, float_type>* alpha_i_k(pair<int, float_type>* alpha_i, int i, int act_indexj){
                int size_alphai = size_alpha[i];
                int index_alpha = hashindices[act_indexj] & (size_alphai - 1);
                hash_bottom++; hash_top++;
                while (alpha_i[index_alpha].first != -1 && alpha_i[index_alpha].first != act_indexj){
                        index_alpha++;
                        if (index_alpha == size_alphai)
                                index_alpha = 0;
                        hash_top++;
                }
                //if (alpha_i[index_alpha].first == act_indexj) hash_top++;
                return &(alpha_i[index_alpha]);
        }

        inline pair<int, float_type>* v_j_k(pair<int, float_type>* vj, int j, int act_indexj){
                int size_vj = size_v[j];
                int index_v = hashindices[act_indexj] & (size_vj - 1);
                hash_bottom++; hash_top++;
                while (vj[index_v].first != -1 && vj[index_v].first != act_indexj){
                        index_v++;
                        if (index_v == size_vj)
                                index_v = 0;
                        hash_top++;
                }
                //if (vj[index_v].first == act_indexj)  hash_top++;
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
		v = new float_type*[D]; //w = prox(v);
		for(int j=0;j<D;j++){
			v[j] = new float_type[K];
			for(int k=0;k<K;k++){
				v[j][k] = 0.0;
			}
		}
		#endif
		//initialize non-zero index array w
		w_nnz_index = new vector<int>**[D];
		for(int j=0;j<D;j++){
			w_nnz_index[j] = new vector<int>*[split_up_rate];
			for(int S=0;S < split_up_rate; S++){
				w_nnz_index[j][S] = new vector<int>();
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
                                        /*int index_alpha = hashindices[act_indexj] & size_alphai0;
                                        //hash_top++; hash_bottom++;
                                        if (alpha_i[index_alpha].first != act_indexj)
                                        while (alpha_i[index_alpha].first != act_indexj && alpha_i[index_alpha].first != -1){
                                                index_alpha++;
                                                index_alpha &= size_alphai0;
                                                //hash_top++;
                                        }*/
                                        find_index(alpha_i, index_alpha, act_indexj, size_alphai0);
                                        alpha_i_k[j] = alpha_i_new[act_indexj] - alpha_i[index_alpha].second;
                                        //act_hashindex[j] = hashindices[act_indexj] & size_alphai0;
                                }	
				#endif
				for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){	
					int J = it->first; 
					float_type f_val = it->second;
					auto vj = v[J];
					#ifdef USING_HASHVEC
					//pair<int, float_type>* vj = v[J];
					int size_vj = size_v[J];
					int util_vj = util_v[J];
					int size_vj0 = size_vj - 1;
					for (int j = 0; j < act_k_index[i].size(); j++){
						int k = act_k_index[i][j];
						//find_index(alpha_i, index_alpha, k, size_alphai0);
						float_type delta_alpha = alpha_i_k[j];
						if( fabs(delta_alpha) < 1e-12 )
							continue;
						//update v
						int S = loc(k); // location of k in [split_up_rate]
						find_index(vj, index_v, k, size_vj0);
						float_type vjk_old = vj[index_v].second;
						float_type vjk = vjk_old + f_val*delta_alpha;
						vj[index_v].second = vjk;
						if (vj[index_v].first == -1){
							vj[index_v].first = k;
							if ((++util_v[J]) > size_vj * UPPER_UTIL_RATE){
								resize(vj, v[J], size_v[J], size_vj, size_vj0, util_v[J]);
							}
						}
						float_type wjk = prox_l1(vjk, lambda);
						float_type wjk_old = prox_l1(vjk_old, lambda);
						if ( wjk_old != wjk ){
                                                        if(  wjk_old == 0.0  ){
                                                                w_nnz_index[J][S]->push_back(k);
                                                        }
                                                }
					}	
					#else
					//float_type* vj = v[J];
					for(int j = 0; j < act_k_index[i].size(); j++){
						int k = act_k_index[i][j];
						float_type delta_alpha = (alpha_i_new[k]-alpha_i[k]);
						if( fabs(delta_alpha) < 1e-12 )
							continue;
						//update v
						int S = loc(k); // location of k in [split_up_rate]
						float_type vjk_old = vj[k];
						float_type vjk = vjk_old + f_val*delta_alpha;
						vj[k] = vjk;
						float_type wjk = prox_l1(vjk, lambda);
						float_type wjk_old = prox_l1(vjk_old, lambda);
						if ( wjk_old != wjk ){
                                                        if(  wjk_old == 0.0  ){
                                                                w_nnz_index[J][S]->push_back(k);
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
					find_index(alpha_i, index_alpha, k, size_alphai0);
					if (fabs(alpha_i_new[k]) > 1e-12){
						alpha_i[index_alpha] = make_pair(k, alpha_i_new[k]);
						/*cout << "alpha outside " << size_alphai*UPPER_UTIL_RATE << " " << util_alpha[i] << endl;
						int cc = 0;
						for (int ii = 0; ii < size_alphai; ii++){
							if (alpha_i[ii].first != -1){
								cc++;
							}
						}
						cout << "alphai " << cc << " " << util_alpha[i] << endl;
						assert(cc == util_alpha[i]);*/
						if ((++util_alpha[i]) > size_alphai * UPPER_UTIL_RATE){
							//cout << "success alpha" << endl;
							resize(alpha_i, alpha[i], size_alpha[i], size_alphai, size_alphai0, util_alpha[i]);
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
			
			if( iter % 1 == 0 ){
				//cerr << "." ;
				cerr << "i=" << iter << "\t";
				int nnz_a_i = 0;
				for(int i=0;i<N;i++){
					nnz_a_i += act_k_index[i].size();	
				}
				cerr << "nnz_a_i="<< ((float_type)nnz_a_i/N) << "  \t";
				int nnz_w_j = 0;
				for(int j=0;j<D;j++){
					for(int S=0;S < split_up_rate; S++){
						nnz_w_j += w_nnz_index[j][S]->size();
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
		double w_1norm = 0.0;
		int jk=0;
		for(int j=0;j<D;j++){
			for(int S=0;S<split_up_rate;S++){
				for (vector<int>::iterator it=w_nnz_index[j][S]->begin(); it!=w_nnz_index[j][S]->end(); it++){
					#ifdef USING_HASHVEC
					int index_v = 0;
					find_index(v[j], index_v, *it, size_v[j]-1);
					float_type wjk = prox_l1(v[j][index_v].second, lambda);
					#else
					float_type wjk = prox_l1(v[j][*it], lambda);
					#endif
					d_obj += wjk*wjk;//W[j][*it]*W[j][*it];
					w_1norm += fabs(wjk);
				}
				nnz_w+=w_nnz_index[j][S]->size();
			}
		}
		/*d_obj/=2.0;
		for(int i=0;i<N;i++){
			Labels* yi = &(labels->at(i));
			for(int k=0;k<K;k++){
				#ifdef USING_HASHVEC
				int index_alpha = 0;
				find_index(alpha[i], index_alpha, k, size_alpha[i] - 1);
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
		cerr << "nSV=" << nSV << " (NK=" << N*K << ")"<< endl;*/
		cerr << "nnz_w=" << nnz_w << " (DK=" << D*K << ")" << endl;
		cerr << "w_1norm=" << w_1norm << endl;
		cerr << "train time=" << endtime-starttime << endl;
		cerr << "search time=" << search_time << endl;
		cerr << "subsolve time=" << subsolve_time << endl;
		cerr << "maintain time=" << maintain_time << endl;
		//delete algorithm-specific variables
		delete[] alpha_i_new;
		delete[] act_k_index;
		for(int i=0;i<N;i++)
			delete[] alpha[i];
		delete[] alpha;
		delete[] Q_diag;
		delete[] prod;
		delete cdf_sum;
		w_temp = new HashVec*[D];
		for(int j = 0; j < D; j++){
			w_temp[j] = new HashVec();
			for(int S=0;S<split_up_rate;S++){
				for (vector<int>::iterator it=w_nnz_index[j][S]->begin(); it!=w_nnz_index[j][S]->end(); it++){
					int k = *it;
					#ifdef USING_HASHVEC
					int index_v = 0;
					find_index(v[j], index_v, k, size_v[j]-1);
					w_temp[j]->insert(make_pair(k, prox_l1(v[j][index_v].second, lambda)));
					#else
					w_temp[j]->insert(make_pair(k, prox_l1(v[j][k], lambda)));
					#endif
				}
			}
		}
		for(int j=0;j<D;j++)
			delete[] v[j];
		delete[] v;
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
                        find_index(alpha_i, index_alpha, p, size_alphai0);
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
			auto vj = v[fea_ind];
			#ifdef USING_HASHVEC
                        //pair<int, float_type>* vj = v[fea_ind];
                        int size_vj = size_v[fea_ind];
                        int size_vj0 = size_vj - 1;
                        int index_v = 0;
                        #else
                        //float_type* vj = v[fea_ind];
                        #endif
			for(int i = 0; i < n; i++){
				int k = act_index_b[i];
				#ifdef USING_HASHVEC
                                find_index(vj, index_v, k, size_vj0);
                                float_type vjk = vj[index_v].second;
                                #else
                                float_type vjk = vj[k];
                                #endif
                                if (fabs(vjk) > lambda){
                                        if (vjk > 0)
                                                b[i] += (vjk-lambda)*fea_val;
                                        else
                                                b[i] += (vjk+lambda)*fea_val;
                                }
			}
			for(int j = 0; j < m; j++){
				int k = act_index_c[j];
				#ifdef USING_HASHVEC
                                find_index(vj, index_v, k, size_vj0);
                                float_type vjk = vj[index_v].second;
                                #else
                                float_type vjk = vj[k];
                                #endif
				if( fabs(vjk) > lambda ){
					if( vjk > 0 )
						c[j] -= (vjk-lambda)*fea_val;
					else
						c[j] -= (vjk+lambda)*fea_val;
				}
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
			vector<int>* wj = w_nnz_index[j][S];
			int k = 0, ind = 0;
			float_type wjk = 0.0;
			auto vj = v[j];
                        for(vector<int>::iterator it2 = wj->begin(); it2<wj->end(); it2++ ){
				k = *it2;
				#ifdef USING_HASHVEC
				int index_v = 0;
				find_index(vj, index_v, k, size_v[j] - 1);
				wjk = prox_l1(vj[index_v].second, lambda);
				#else
				wjk = prox_l1(vj[k], lambda);
				#endif
				if (wjk == 0.0){
					*it2=*(wj->end()-1);
					wj->erase(wj->end()-1);
					it2--;
					continue;
				}
                                prod[k] += wjk * xij;
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
        }

	void search_active_i_uniform(int i, vector<int>& act_k_index){
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
	vector<float_type>* cdf_sum;
	HashVec** w_temp;
	vector<int>*** w_nnz_index;
	int max_iter;
	vector<int>* k_index;
		
	//sampling 
	bool using_importance_sampling;
	int max_select;
	int speed_up_rate, split_up_rate;	
	float_type* prod;
	
};
