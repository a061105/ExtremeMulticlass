#include "util.h"
#include "multi.h"
#include "newHash.h"
#include <iomanip>
#include <cassert>
#define loc(k) k*split_up_rate/K

extern double overall_time;

class SplitOracleActBCD{
	
	public:
	SplitOracleActBCD(Param* param){
		train = param->train;
		heldoutEval = param->heldoutEval;	
		early_terminate = param->early_terminate;
		data = &(train->data);
		labels = &(train->labels);
		lambda = param->lambda;
		C = param->C;
		N = train->N;
		D = train->D;
		K = train->K;
		dump_model = param->dump_model;
		if (param->dump_model){
			modelFname =string( param->modelFname);
		}

		//a random permutation
		hashfunc = new HashClass(K);
		hashindices = hashfunc->hashindices;
		
		//compute useful statistics and l_1 norm of every feature x_i
		nnz_a_i = 0.0; d = 0.0;
		cdf_sum = new vector<Float>();
		for(int i = 0; i < N; i++){
			SparseVec* xi = data->at(i);
			Float _cdf = 0.0;
			nnz_a_i += labels->at(i).size();
			d += xi->size();
			for (SparseVec::iterator it = xi->begin(); it < xi->end(); it++){
				_cdf += fabs(it->second);
			}
			cdf_sum->push_back(_cdf);
		}
		nnz_a_i /= N; d /= N;
		max_iter = param->max_iter;

		//setting up for sampling oracle
		speed_up_rate = param->speed_up_rate;
		split_up_rate = param->split_up_rate;
		if( speed_up_rate==-1 )
			speed_up_rate = ceil( min(5.0*D*K/nnz(train->data)/C/log((Float)K), d/10.0) );
		cerr << "lambda=" << lambda << ", C=" << C << endl;
		using_importance_sampling = param->using_importance_sampling;
		if (using_importance_sampling){
			cerr << "using importance sampling" << ", speed up rate=" << speed_up_rate << endl;
		} else {
			cerr << "using uniform sampling" << ", speed up rate=" << speed_up_rate << endl;
		}

		//number of variables added to active set in each iteration.
		max_select = param->max_select;
		if (max_select == -1){
			int avg_label = 0;
			for (int i = 0; i < N; i++){
				avg_label += labels->at(i).size();
			}
			avg_label /= N;
			if (avg_label < 1)
				avg_label = 1;
			max_select = avg_label;
		}
		//global cache
		prod = new Float[K];	
		prod_cache = new Float[K];
		inside = new bool[K];
		inside_index = new bool[K];
		memset(prod, 0.0, sizeof(Float)*K);
		memset(prod_cache, 0.0, sizeof(Float)*K);
		memset(inside, false, sizeof(bool)*K);
		memset(inside_index, false, sizeof(bool)*K);
	}
	
	~SplitOracleActBCD(){
		for(int j=0;j<D;j++)
			delete[] v[j];
		delete[] v;
		
		for(int j=0;j<D;j++)
			delete[] w[j];
		delete[] w;
		
		//delete global cache
		delete[] inside;
		delete[] inside_index;
		delete[] prod;
		delete[] prod_cache;
		
		#ifdef USING_HASHVEC
		delete[] size_v;
		delete[] util_v;
		delete[] size_w;
		#endif

		delete[] act_k_index;
		delete[] hashindices;
		delete[] non_split_index;	
		delete[] w_hash_nnz_index;
	}

	Model* solve(){
		//initialize alpha and v ( s.t. v = X^Talpha )
		
		//for storing best model
		non_split_index = new vector<int>[D];
		#ifdef USING_HASHVEC
                v = new pair<int, pair<Float, Float>>*[D];
                size_v = new int[D];
                util_v = new int[D];
                memset(util_v, 0, D*sizeof(int));
                for (int j = 0; j < D; j++){
                        size_v[j] = INIT_SIZE;
                        v[j] = new pair<int, pair<Float,Float>>[size_v[j]];
                        for(int k = 0; k < size_v[j]; k++){
                                v[j][k] = make_pair(-1, make_pair(0.0, 0.0));
                        }
                }
		//for storing best model w
		size_w = new int[D];
		w = new pair<int, Float>*[D];
		for (int j = 0; j < D; j++){
			w[j] = new pair<int, Float>[1];
		}
		#else
		v = new pair<Float, Float>*[D]; //w = prox(v);
		for(int j=0;j<D;j++){
			v[j] = new pair<Float, Float>[K];
			for(int k=0;k<K;k++){
				v[j][k] = make_pair(0.0, 0.0);
			}
		}
		//for storing best model w
		w = new Float*[D];
		for (int j = 0; j < D; j++){
			w[j] = new Float[K];
			memset(w[j], 0.0, sizeof(Float)*K);
		}
		#endif
		//initialize non-zero index array w
		w_hash_nnz_index = new vector<int>*[D];
		for(int j=0;j<D;j++){
			w_hash_nnz_index[j] = new vector<int>[split_up_rate];
			for(int S=0;S < split_up_rate; S++){
				w_hash_nnz_index[j][S].clear();
			}
		}
		//initialize Q_diag (Q=X*X') for the diagonal Hessian of each i-th subproblem
		Q_diag = new Float[N];
		for(int i=0;i<N;i++){
			SparseVec* ins = data->at(i);
			Float sq_sum = 0.0;
			for(SparseVec::iterator it=ins->begin(); it!=ins->end(); it++)
				sq_sum += it->second*it->second;
			Q_diag[i] = sq_sum;
		}
		//indexes for permutation of [N]
		int* index = new int[N];
		for(int i=0;i<N;i++)
			index[i] = i;
		//initialize active set out of [K] for each sample i
		act_k_index = new vector<pair<int, Float>>[N];
		for(int i=0;i<N;i++){
			Labels* yi = &(labels->at(i));
			for (Labels::iterator it = yi->begin(); it < yi->end(); it++){
				act_k_index[i].push_back(make_pair(*it, 0.0));
			}
		}
		//for storing best model
		best_act_k_index = NULL;
		
		//main loop
		int terminate_countdown = 0;
		double search_time=0.0, subsolve_time=0.0, maintain_time=0.0;
		double last_search_time = 0.0, last_subsolve_time = 0.0, last_maintain_time = 0.0;
		Float* alpha_i_new = new Float[K];
		iter = 0;
		best_heldout_acc = -1.0; best_model = NULL;
		while( iter < max_iter ){
			
			random_shuffle( index, index+N );
			for(int r=0;r<N;r++){
				
				int i = index[r];
				SparseVec* x_i = data->at(i);
				Labels* yi = &(labels->at(i));
				
				#ifdef USING_HASHVEC
				int index_alpha = 0, index_v = 0;
				#endif
				//search active variable
				search_time -= omp_get_wtime();
				if (using_importance_sampling && x_i->size() >= speed_up_rate )
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
				maintain_time -= omp_get_wtime();
				Float* delta_alpha_ik = new Float[act_k_index[i].size()];
				int ind = 0;
				for(vector<pair<int, Float>>::iterator it = act_k_index[i].begin(); it != act_k_index[i].end(); it++){
                                        delta_alpha_ik[ind++] = alpha_i_new[it->first] - it->second;
				}
				for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
					int J = it->first; 
					Float f_val = it->second;
					vector<int>* wJ = w_hash_nnz_index[J];
					#ifdef USING_HASHVEC
					pair<int, pair<Float, Float>>* vj = v[J];
					int size_vj = size_v[J];
					int util_vj = util_v[J];
					int size_vj0 = size_vj - 1;
					ind = 0;
					for (vector<pair<int, Float>>::iterator it2 = act_k_index[i].begin(); it2 != act_k_index[i].end(); it2++){
						int k = it2->first;
						Float delta_alpha = delta_alpha_ik[ind++];
						if( fabs(delta_alpha) < EPS )
							continue;
						//update v, w
						find_index(vj, index_v, k, size_vj0, hashindices);
						Float vjk = vj[index_v].second.first + f_val*delta_alpha;
						Float wjk_old = vj[index_v].second.second;
						Float wjk = prox_l1(vjk, lambda);
						vj[index_v].second = make_pair(vjk, wjk);
						if (vj[index_v].first == -1){
							vj[index_v].first = k;
							if ((++util_v[J]) > size_vj * UPPER_UTIL_RATE){
								resize(vj, v[J], size_v[J], size_vj, size_vj0, util_v[J], hashindices);
							}
						}
						if ( wjk_old != wjk ){
							if (fabs(wjk_old) < EPS){
								wJ[loc(k)].push_back(k);
							}
                                                }
					}
					#else
					pair<Float, Float>* vj = v[J];
					ind = 0;
					for(vector<pair<int, Float>>::iterator it2 = act_k_index[i].begin(); it2 < act_k_index[i].end(); it2++){
						int k = it2->first;
						Float delta_alpha = delta_alpha_ik[ind++];
						if( fabs(delta_alpha) < EPS )
							continue;
						//update v, w
						pair<Float, Float> vjk_wjk = vj[k];
						Float vjk = vjk_wjk.first + f_val*delta_alpha;
						Float wjk = prox_l1(vjk, lambda);
						Float wjk_old = vjk_wjk.second;
						vj[k] = make_pair(vjk, wjk);
						if ( wjk_old != wjk ){
							if (fabs(wjk_old) < EPS){
								wJ[loc(k)].push_back(k);
							}
                                                }
					}
					#endif
				}
				delete[] delta_alpha_ik;
				//update alpha
				bool has_zero=0;
				for(vector<pair<int, Float>>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
					int k = it->first;
					it->second = alpha_i_new[k];
					has_zero |= (fabs(it->second)<EPS);
				}
				//shrink act_k_index
				if( has_zero ){
					vector<pair<int, Float>> tmp_vec;
					tmp_vec.reserve(act_k_index[i].size());
					for(vector<pair<int, Float>>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
						int k = it->first;
						if( fabs(it->second) > EPS || find(yi->begin(), yi->end(), k)!=yi->end() ){
							tmp_vec.push_back(make_pair(k, it->second));
						}
					}
					act_k_index[i] = tmp_vec;
				}
				maintain_time += omp_get_wtime();
			}

			cerr << "i=" << iter << "\t" ;
			nnz_a_i = 0.0;
			for(int i=0;i<N;i++){
				nnz_a_i += act_k_index[i].size();	
			}
			nnz_a_i /= N;
			cerr << "nnz_a_i="<< (nnz_a_i) << "\t";
			nnz_w_j = 0.0;
			for(int j=0;j<D;j++){
				for(int S=0;S < split_up_rate; S++){
					nnz_w_j += w_hash_nnz_index[j][S].size(); //util_w[j][S];
					
				}
			}
			nnz_w_j /= D;
			cerr << "nnz_w_j=" << (nnz_w_j) << "\t";
			cerr << "search=" << search_time-last_search_time << "\t";
			cerr << "subsolve=" << subsolve_time-last_subsolve_time << "\t";
			cerr << "maintain=" << maintain_time-last_maintain_time << "\t";
			if (search_time - last_search_time > (subsolve_time-last_subsolve_time + maintain_time - last_maintain_time)*2){
				max_select *= 2;
			}
			if (max_select > 100){
				max_select = 100;
			}
			last_search_time = search_time;
			last_maintain_time = maintain_time;
			last_subsolve_time = subsolve_time;
			overall_time += omp_get_wtime();
			cerr << "dual_obj=" << dual_obj() << "\t";
			//early terminate: if heldout_test_accuracy does not increase in last <early_terminate> iterations, stop!	
			if( heldoutEval != NULL){
				#ifdef USING_HASHVEC
				Float heldout_test_acc = heldoutEval->calcAcc(v, size_v, w_hash_nnz_index, hashindices, split_up_rate);
				#else
				Float heldout_test_acc = heldoutEval->calcAcc(v, w_hash_nnz_index, split_up_rate);
				#endif
				cerr << "heldout Acc=" << heldout_test_acc << " ";
				if ( heldout_test_acc > best_heldout_acc){
					best_heldout_acc = heldout_test_acc;
					store_best_model();
					if (dump_model){
						string name = modelFname + "." + to_string(iter);
						char* fname = new char[name.length()+1];
						strcpy(fname, name.c_str());
						cerr << ", dump_model_file=" << fname;
						best_model->writeModel(fname);
						delete fname;
					}
					terminate_countdown = 0;
				} else {
					cerr << "(" << (++terminate_countdown) << "/" << early_terminate << ")";
					if (terminate_countdown == early_terminate){
						overall_time -= omp_get_wtime();
						break;
					}
				}
			}
			cerr << endl;
			
			overall_time -= omp_get_wtime();
			iter++;
		}
		cerr << endl;

		//recover act_k_index to the best state so far
		//This is because act_k_index is not a part of model, but we might need to use act_k_index possibly in Post Solve
		if (best_act_k_index != NULL){
			for (int i = 0; i < N; i++){
				act_k_index[i] = best_act_k_index[i];
			}
		}
		if (best_model == NULL){
			store_best_model();
		}
	
		//computing heldout accuracy 	
		cerr << "train time=" << (overall_time + omp_get_wtime()) << endl;
		cerr << "search time=" << search_time << endl;
		cerr << "subsolve time=" << subsolve_time << endl;
		cerr << "maintain time=" << maintain_time << endl;
		//delete algorithm-specific variables
		delete[] alpha_i_new;
		delete[] Q_diag;
		delete cdf_sum;
		delete[] index;
		return best_model;
	}
	
	//compute 1/2 \|w\|_2^2 + \sum_{i,k: k \not \in y_i} alpha_{i, k}
	Float dual_obj(){
		Float dual_obj = 0.0;
		memset(inside, false, sizeof(bool)*K);
		for (int J = 0; J < D; J++){
			vector<int>* wJ = w_hash_nnz_index[J];
			#ifdef USING_HASHVEC
			pair<int, pair<Float, Float>>* vj = v[J];
			int size_vj = size_v[J];
			int util_vj = util_v[J];
			int size_vj0 = size_vj - 1;
			int index_v = -1;
			#else
			pair<Float, Float>* vj = v[J];
			#endif
			
			for (int S = 0; S < split_up_rate; S++){
				for (vector<int>::iterator it = wJ[S].begin(); it != wJ[S].end(); it++){
					int k = *it;
					if (inside[k]){
						continue;
					}
					inside[k] = true;
					#ifdef USING_HASHVEC
					find_index(vj, index_v, k, size_vj0, hashindices);
					Float wjk = vj[index_v].second.second;
					#else
					Float wjk = vj[k].second;
					#endif
					dual_obj += wjk*wjk;
				}
				for (vector<int>::iterator it = wJ[S].begin(); it != wJ[S].end(); it++){
					int k = *it;
					inside[k] = false;
				}
			}
		}
		dual_obj /= 2.0;
		for (int i = 0; i < N; i++){
			vector<pair<int, Float>>& act_index = act_k_index[i];
			Labels* yi = &(labels->at(i));
			for (vector<pair<int, Float>>::iterator it = act_index.begin(); it != act_index.end(); it++){
				int k = it->first;
				Float alpha_ik = it->second;
				if (find(yi->begin(), yi->end(), k) == yi->end()){
					dual_obj += alpha_ik;
				}
			}
		}
		return dual_obj;
	}

	void subSolve(int I, vector<pair<int, Float>>& act_k_index, Float* alpha_i_new){	
			
		Labels* yi = &(labels->at(I));
		int m = yi->size(), n = act_k_index.size() - m;
		Float* b = new Float[n];
		Float* c = new Float[m];
		int* act_index_b = new int[n];
		int* act_index_c = new int[m];
		
		SparseVec* x_i = data->at(I);
		Float A = Q_diag[I];
		int i = 0, j = 0;
		for(vector<pair<int, Float>>::iterator it = act_k_index.begin(); it != act_k_index.end(); it++){
			int k = it->first;
			Float alpha_ik = it->second;
                        if( find(yi->begin(), yi->end(), k) == yi->end() ){
                                b[i] = 1.0 - A*alpha_ik;
                                act_index_b[i++] = k;
                        }else{
                                c[j] = A*alpha_ik;
                                act_index_c[j++] = k;
                        }
		}

		for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
			int fea_ind = it->first;
			Float fea_val = it->second;
			#ifdef USING_HASHVEC
                        pair<int, pair<Float, Float>>* vj = v[fea_ind];
                        int size_vj = size_v[fea_ind];
                        int size_vj0 = size_vj - 1;
                        int index_v = 0;
                        #else
                        pair<Float, Float>* vj = v[fea_ind];
                        #endif
			for(int i = 0; i < n; i++){
				#ifdef USING_HASHVEC
                                find_index(vj, index_v, act_index_b[i], size_vj0, hashindices);
                                Float wjk = vj[index_v].second.second;
                                b[i] += wjk*fea_val;
                                #else
                                b[i] += vj[act_index_b[i]].second*fea_val;
                                #endif
			}
			for(int j = 0; j < m; j++){
				#ifdef USING_HASHVEC
                                find_index(vj, index_v, act_index_c[j], size_vj0, hashindices);
                                Float wjk = vj[index_v].second.second;
                                c[j] -= wjk*fea_val;
                                #else
                                c[j] -= vj[act_index_c[j]].second*fea_val;
				#endif
			}
		}
		for (int i = 0; i < n; i++){
			b[i] /= A;
		}
		for (int j = 0; j < m; j++){
			c[j] /= A;
		}
			
		Float* x = new Float[n];
		Float* y = new Float[m];
		solve_bi_simplex(n, m, b, c, C, x, y);
		for(int i = 0; i < n; i++){
			int k = act_index_b[i];
			alpha_i_new[k] = -x[i];
		}
		for(int j = 0; j < m; j++){
                        int k = act_index_c[j];
			alpha_i_new[k] = y[j];
                }

		delete[] x; delete[] y;
		delete[] b; delete[] c;
		delete[] act_index_b; delete[] act_index_c;
	}
	
	//search with importance sampling	
	void search_active_i_importance( int i, vector<pair<int, Float>>& act_k_index ){
		//prod_cache should be all zero
		
		//select one area from {0, ..., split_up_rate-1}
		int S = rand()%split_up_rate;
		
		//compute <xi,wk> for k in the area just chosen
                Labels* yi = &(labels->at(i));
                vector<int> check_indices;
		SparseVec* xi = data->at(i);
		int nnz = xi->size();
		for(vector<pair<int, Float>>::iterator it = act_k_index.begin(); it != act_k_index.end(); it++){
			prod_cache[it->first] = -INFI;
		}
		for (Labels::iterator it = yi->begin(); it != yi->end(); it++){
			prod_cache[*it] = -INFI;
		}
		int n = nnz/speed_up_rate;
		Float th = -n/(1.0*nnz);
		vector<Float> rand_nums;
		for (int tt = 0; tt < n; tt++){
			rand_nums.push_back(((Float)rand()/(RAND_MAX)));
		}
		sort(rand_nums.begin(), rand_nums.end()); 
		int* max_indices = new int[max_select+1];
		for(int ind = 0; ind <= max_select; ind++){
			max_indices[ind] = -1;
		}
		SparseVec::iterator current_index = xi->begin();
		Float current_sum = current_index->second;
		vector<Float>::iterator current_rand_index = rand_nums.begin();
		Float cdf_sumi = cdf_sum->at(i);
		while (current_rand_index < rand_nums.end()){
			while (current_sum < (*current_rand_index)*cdf_sumi){
				current_index++;
				current_sum += current_index->second;
			}
			Float xij = 0.0;
			while (current_rand_index < rand_nums.end() && current_sum >= (*current_rand_index)*cdf_sumi ){
				xij = xij + 1.0;
				current_rand_index++;
			}
                        xij *= cdf_sumi*((current_index->second > 0.0)?1:(-1));
			int j = current_index->first;
			vector<int>& wjS = w_hash_nnz_index[j][S];
			if (wjS.size() == 0) 
				continue;
			Float wjk = 0.0;
			#ifdef USING_HASHVEC
			int size_vj0 = size_v[j] - 1;
			pair<int, pair<Float, Float>>* vj = v[j];
			#else
			pair<Float, Float>* vj = v[j];
			#endif
			for(vector<int>::iterator it2 = wjS.begin(); it2!=wjS.end(); it2++ ){
				int k = *(it2);
				#ifdef USING_HASHVEC
				int index_v = 0;
				find_index(vj, index_v, k, size_vj0, hashindices);
				wjk = vj[index_v].second.second;
				#else
				wjk = vj[k].second;
				#endif
				if (wjk == 0.0 || inside[k]){
                                        *it2=*(wjS.end()-1);
                                        wjS.erase(wjS.end()-1);
                                        it2--;
                                        continue;
                                }
				if (!inside_index[k]){
					check_indices.push_back(k);
					inside_index[k] = true;
				}
				inside[k] = true;
                                prod_cache[k] += wjk * xij;
			}
                        for(vector<int>::iterator it2 = wjS.begin(); it2!=wjS.end(); it2++ ){
				inside[*it2] = false;
			}
                }
		
		for (vector<int>::iterator it = check_indices.begin(); it != check_indices.end(); it++){
			int k = *it;
			inside_index[k] = false;
			if (prod_cache[k] >= th)
				update_max_indices(max_indices, prod_cache, k, max_select);
		}
		for (int j = 0; j < max_select; j++){
			if (max_indices[j] != -1 && prod_cache[max_indices[j]] > 0.0) 
				continue;
			for (int r = 0; r < K; r++){
				int k = hashindices[r];
				if (prod_cache[k] == 0.0){
					if (update_max_indices(max_indices, prod_cache, k, max_select)){
						break;
					}
				}
			}
		}
		for(int ind = 0; ind < max_select; ind++){
			if (max_indices[ind] != -1 && prod_cache[max_indices[ind]] > th){
				act_k_index.push_back(make_pair(max_indices[ind], 0.0));
			}
		}
		
		//reset prod_cache to all zero
		for (vector<int>::iterator it = check_indices.begin(); it != check_indices.end(); it++){
			prod_cache[*it] = 0.0;
		}
		for(vector<pair<int, Float>>::iterator it = act_k_index.begin(); it != act_k_index.end(); it++){
			prod_cache[it->first] = 0.0;
		}
		for (Labels::iterator it = yi->begin(); it != yi->end(); it++){
			prod_cache[*it] = 0.0;
		}
		
		delete[] max_indices;
        }

	//searching with uniform sampling
	void search_active_i_uniform(int i, vector<pair<int, Float>>& act_k_index){	
		//prod_cache should be all zero
		//select one area from {0, ..., split_up_rate-1}
		int S = rand()%split_up_rate;

                //compute <xi,wk> for k=1...K
                Labels* yi = &(labels->at(i));
                vector<int> check_indices;
                SparseVec* xi = data->at(i);
		int nnz = xi->size();
		for(vector<pair<int, Float>>::iterator it = act_k_index.begin(); it != act_k_index.end(); it++){
			prod_cache[it->first] = -INFI;
		}
		for (Labels::iterator it = yi->begin(); it < yi->end(); it++){
			prod_cache[*it] = -INFI;
		}
		int n = nnz/speed_up_rate;
		if (nnz < speed_up_rate)
			n = nnz;
		Float th = -n/(1.0*nnz);
		int* max_indices = new int[max_select+1];
		for(int ind = 0; ind <= max_select; ind++){
			max_indices[ind] = -1;
		}
		random_shuffle(xi->begin(), xi->end());
		for (SparseVec::iterator current_index = xi->begin(); current_index < xi->begin() + n; current_index++){
			Float xij = current_index->second;
			int j = current_index->first;
			vector<int>& wjS = w_hash_nnz_index[j][S];
			if (wjS.size() == 0) continue;
			int k = 0, ind = 0;
			#ifdef USING_HASHVEC
			int size_vj0 = size_v[j] - 1;
			#endif
			Float wjk = 0.0;
			auto vj = v[j];
                        for(vector<int>::iterator it2 = wjS.begin(); it2!=wjS.end(); it2++ ){
				k = *(it2);
				#ifdef USING_HASHVEC
				int index_v = 0;
				find_index(vj, index_v, k, size_vj0, hashindices);
				wjk = vj[index_v].second.second;
				#else
				wjk = vj[k].second;
				#endif
				if (wjk == 0.0 || inside[k]){
                                        *it2=*(wjS.end()-1); 
                                        wjS.erase(wjS.end()-1); 
                                        it2--;
                                        continue;
                                }
				if (!inside_index[k]){
					check_indices.push_back(k);
					inside_index[k] = true;
				}
				inside[k] = true;
                                prod_cache[k] += wjk * xij;
			}
			for (vector<int>::iterator it2 = wjS.begin(); it2 != wjS.end(); it2++){
				inside[*it2] = false;
			}
                }
		for (vector<int>::iterator it = check_indices.begin(); it != check_indices.end(); it++){
			int k = *it;
			inside_index[k] = false;
			if (prod_cache[k] >= th)
				update_max_indices(max_indices, prod_cache, k, max_select);
		}
		for (int j = 0; j < max_select; j++){
			if (max_indices[j] != -1 && prod_cache[max_indices[j]] > 0.0) 
				continue;
			for (int r = 0; r < K; r++){
				int k = hashindices[r];
				if (prod_cache[k] == 0){
					if (update_max_indices(max_indices, prod_cache, k, max_select)){
						break;
					}
				}
			}
		}
		for(int ind = 0; ind < max_select; ind++){
			if (max_indices[ind] != -1 && prod_cache[max_indices[ind]] > th){
				act_k_index.push_back(make_pair(max_indices[ind], 0.0));
			}
		}
		
		//reset prod_cache to all zero
		for (vector<int>::iterator it = check_indices.begin(); it != check_indices.end(); it++){
			prod_cache[*it] = 0.0;
		}
		for(vector<pair<int, Float>>::iterator it = act_k_index.begin(); it != act_k_index.end(); it++){
			prod_cache[it->first] = 0.0;
		}
		for (Labels::iterator it = yi->begin(); it != yi->end(); it++){
			prod_cache[*it] = 0.0;
		}
		delete[] max_indices;
	}

	//store the best model as well as necessary indices
	void store_best_model(){
		#ifdef USING_HASHVEC
		memset(inside, false, sizeof(bool)*K);
		for (int j = 0; j < D; j++){
			size_w[j] = 1;
			int total_size = 0;
			for (int S = 0; S < split_up_rate; S++){
				total_size+=w_hash_nnz_index[j][S].size();
			}
			while (size_w[j] * UPPER_UTIL_RATE < total_size)
				size_w[j] *= 2;
			delete[] w[j];
			w[j] = new pair<int, Float>[size_w[j]];
			non_split_index[j].clear();
			for(int it = 0; it < size_w[j]; it++)
				w[j][it] = make_pair(-1, 0.0);
			memset(inside, false, sizeof(bool)*K);
			for(int S=0;S<split_up_rate;S++){
				for (vector<int>::iterator it=w_hash_nnz_index[j][S].begin(); it!=w_hash_nnz_index[j][S].end(); it++){
					int k = *it;
					int index_v = 0;
					find_index(v[j], index_v, k, size_v[j]-1, hashindices);
					if (fabs(v[j][index_v].second.second) > 1e-12 && !inside[k]){
						inside[k] = true;
						int index_w = 0;
						find_index(w[j], index_w, k, size_w[j]-1, hashindices);
						w[j][index_w] = make_pair(k, v[j][index_v].second.second);
						non_split_index[j].push_back(k);
					}
				}
			}
			//recover inside, avoid any complexity related to K
			for (vector<int>::iterator it=non_split_index[j].begin(); it != non_split_index[j].end(); it++){
				int k = *it;
				inside[k] = false;
			}
		}
		best_model = new Model(train, non_split_index, w, size_w, hashindices);
		#else
		for (int j = 0; j < D; j++){
			for (vector<int>::iterator it = non_split_index[j].begin(); it != non_split_index[j].end(); it++){
				w[j][*it] = 0.0;
			}
		}
		memset(inside, false, sizeof(bool)*K);
		for (int j = 0; j < D; j++){
			non_split_index[j].clear();
			pair<Float, Float>* vj = v[j];
			for(int S=0;S<split_up_rate;S++){
				for (vector<int>::iterator it=w_hash_nnz_index[j][S].begin(); it!=w_hash_nnz_index[j][S].end(); it++){
					int k = *it;
					if (fabs(vj[k].second) > EPS && !inside[k]){
						w[j][k] = vj[k].second;
						inside[k] = true;
						non_split_index[j].push_back(k);
					}
				}
			}
			//recover inside, avoid any complexity related to K
			for (vector<int>::iterator it=non_split_index[j].begin(); it != non_split_index[j].end(); it++){
				int k = *it;
				inside[k] = false;
			}
		}
		best_model = new Model(train, non_split_index, w);
		#endif		
		if (best_act_k_index == NULL)
			best_act_k_index = new vector<pair<int, Float>>[N];
		for (int i = 0; i < N; i++)
			best_act_k_index[i] = act_k_index[i];
	}
	
	private:
	
	double best_heldout_acc = -1.0;
	Problem* train;
	HeldoutEval* heldoutEval;
	Float lambda;
	Float C;
	vector<SparseVec*>* data;
	vector<Labels>* labels;
	int D; 
	int N;
	int K;
	Float* Q_diag;
        HashClass* hashfunc;
	vector<Float>* cdf_sum;
	HashVec** w_temp;
	vector<int>** w_hash_nnz_index;
	
	int max_iter;
	vector<int>* k_index;
		
	//sampling 
	bool* inside_index;
	Float* prod_cache;
	bool using_importance_sampling;
	int max_select;
	int speed_up_rate, split_up_rate;
	
	//global cache
	bool* inside;
	Float* prod;

	//heldout options
	int early_terminate;
		
	public:
	
	//useful statistics
	Float nnz_w_j = 1.0;
	Float nnz_a_i = 1.0;
	Float d = 1.0;

	//(index, val) representation of alpha
	vector<pair<int, Float>>* act_k_index;

	//for storing best model
	vector<pair<int, Float>>* best_act_k_index;
	vector<int>* non_split_index;
	Model* best_model = NULL;
	vector<int>** best_w_hash_nnz_index;

	//iterations used so far	
	int iter;
	//a random permutation stored in public
	int* hashindices;
	
	bool dump_model = false;
	string modelFname;
	
	
	#ifdef USING_HASHVEC
	pair<int, Float>** w;
	int* size_w;
	pair<int, pair<Float, Float> >** v;
	pair<int, pair<Float, Float> >** best_v;  
	int* size_v;
        int* util_v;
	#else
	Float** w;
	pair<Float, Float>** v;
	pair<Float, Float>** best_v;
	#endif
};
